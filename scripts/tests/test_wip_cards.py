import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('wip_cards', ROOT / 'scripts/bench/wip-cards.py')
cards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cards)


@pytest.fixture
def entry(tmp_path):
    base = tmp_path / 'model.config'
    base.write_text('MODEL_ID=test/model\nBENCH_SIMULATED=on\n')
    return dict(name='v41-min', family='deepseek_v41', config=str(base), gpus=[0], sparks=['ostrich', 'dodo', 'emu'], set={'SPARK_COUNT': '3', 'ADDR': '0.0.0.0:8500', 'EXPERT_PORT': '19600', 'API_KEY_FILE': '/foreign/key'}, source_matrix='kit')


def make(entry, tmp_path, arm='baseline', instance='test-base'):
    return cards.generate(entry, arm, dict(instance=instance, slot='s'), tmp_path, tmp_path / 'api-key', {}, set(), None, 1)


def test_matrix_private_key_and_complete_lanes(entry, tmp_path):
    job = make(entry, tmp_path)
    matrix = json.loads((Path(job['state']) / 'matrix.json').read_text())
    e = matrix['entries'][0]
    assert 'BENCH_SIMULATED' not in e['set']
    assert 'BENCH_SIMULATED' not in Path(e['config']).read_text()
    assert e['set']['API_KEY_FILE'] == str(tmp_path / 'api-key')
    assert e['set']['ENABLE_BENCH'] == 'on'
    assert e['set']['SPARK_2_HOST'] == 'emu'
    assert e['set']['SPARK_2_LANE_A'] == '10.55.0.3'
    assert e['set']['SPARK_2_LANE_B'] == '10.55.1.3'
    assert e['set']['WIP_INSTANCE'] == 'test-base'
    assert e['run_args'] == ['--wip', 's']
    assert job['simulated']


def test_key_file_created_private_and_never_printed(tmp_path, entry, monkeypatch, capsys):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry])))
    state = tmp_path / 'state'
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix', '--arm', 'base=test:s', '--state', str(state), '--dry-run'])
    assert cards.main() == 0
    key = state / 'api-key'
    assert key.stat().st_mode & 0o777 == 0o600
    assert key.read_text().strip() not in capsys.readouterr().out


def test_parallel_refuses_published_and_overlap(entry, tmp_path):
    a = make(entry, tmp_path)
    b_entry = {**entry, 'name': 'other', 'gpus': [1], 'sparks': ['rhea', 'moa'], 'set': {'SPARK_COUNT': '2', 'ADDR': '0.0.0.0:8501', 'EXPERT_PORT': '19601'}}
    b = make(b_entry, tmp_path, 'candidate', 'test-candidate')
    cards.parallel_safe([a, b], 2)
    a['metadata']['published'] = True
    with pytest.raises(ValueError, match='performance card'):
        cards.parallel_safe([a, b], 2)
    a['metadata'].pop('published')
    b['entry']['gpus'] = [0]
    with pytest.raises(ValueError, match='overlap'):
        cards.parallel_safe([a, b], 2)


def test_parallel_real_perf_refused(entry, tmp_path):
    job = make(entry, tmp_path)
    job['simulated'] = False
    with pytest.raises(ValueError, match='correctness-only'):
        cards.parallel_safe([job], 2)


def test_grouped_smoke_preferred_for_shared_arm(entry, tmp_path):
    a = make(entry, tmp_path)
    b = make({**entry, 'name': 'other'}, tmp_path)
    assert cards.smoke_groups([a, b], 2) == [[a, b]]
    b['arm'] = 'candidate'
    assert cards.smoke_groups([a, b], 2) == [[a], [b]]


def test_rc2_lock_sets_change_only_gpu1_out_of_pool():
    kit = Path.home() / '.cache/cuteafd/builds/release-v2-rc2/kit'
    if not kit.is_dir():
        pytest.skip('RC2 kit not installed')
    for matrix in kit.glob('matrix-*.json'):
        data = json.loads(matrix.read_text())
        for raw in data['entries']:
            e = {**data.get('defaults', {}), **raw}
            gpus, hosts = e.get('gpus', [0]), e.get('sparks', [])
            exclusive = e.get('exclusive', e.get('family') == 'deepseek_v41')
            old = {'sparks.lock'} if 0 in gpus or hosts or exclusive else set()
            new = {'sparks.lock'} if 0 in gpus or set(hosts) & {'ostrich', 'dodo', 'emu', 'kiwi'} or exclusive else set()
            other = {f'gpu{gpu}.lock' for gpu in gpus} | {f'{h}.lock' for h in hosts if h in {'rhea', 'moa'}}
            if exclusive:
                other |= {'gpu0.lock', 'gpu1.lock'}
            if old != new:
                assert gpus == [1] and hosts and set(hosts) <= {'rhea', 'moa'} and not exclusive, e['name']
                assert (old | other) - (new | other) == {'sparks.lock'}
    source = (ROOT / 'rust/crates/cuteafd-bench/src/smoke.rs').read_text()
    assert 'if gpus.contains(&0) || pool || entry.exclusive()' in source
    assert '"ostrich" | "dodo" | "emu" | "kiwi"' in source


def test_smoke_no_outer_lock_and_detached_numeric_exit(entry, tmp_path, monkeypatch):
    job = make(entry, tmp_path)
    command = cards.smoke_command('/driver', ROOT, job)
    assert 'flock' not in command
    seen = []
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **kw: seen.append((cmd, kw)))
    cards.detached(command, tmp_path / 'log', tmp_path / 'exit', {'CUTEAFD_API_KEY': 'secret'})
    cmd, kwargs = seen[0]
    assert cmd[0] == 'setsid'
    assert 'secret' not in str(cmd)
    assert 'printf' in cmd[3] and '$rc' in cmd[3]
    assert kwargs['env']['CUTEAFD_API_KEY'] == 'secret'


def test_prelaunch_foreign_container_refused(entry, tmp_path, monkeypatch):
    job = make(entry, tmp_path)
    monkeypatch.setattr(cards, 'run', lambda cmd: SimpleNamespace(stdout='cuteafd-coordinator-' + job['entry']['set']['INSTANCE'] + '\n'))
    with pytest.raises(ValueError, match='foreign'):
        cards.assert_absent(job)


def test_slot_check_every_host_and_seal(monkeypatch):
    calls = []
    monkeypatch.setattr(cards, 'run', lambda cmd: calls.append(cmd) or SimpleNamespace(stdout='{}'))
    arm = dict(instance='test', slot='s')
    for host in ('raptor', 'rhea', 'moa'):
        cards.slot_check(arm, host)
    assert calls[1][:4] == ['ssh', '-o', 'BatchMode=yes', 'rhea']
    assert calls[2][:4] == ['ssh', '-o', 'BatchMode=yes', 'moa']
    assert all('FINGERPRINT' in str(c) and 'artifact_manifest_sha256' in str(c) for c in calls)
    monkeypatch.setattr(cards, 'run', lambda cmd: (_ for _ in ()).throw(subprocess.CalledProcessError(1, cmd, stderr='missing META')))
    with pytest.raises(ValueError, match='rhea: missing or unsealed'):
        cards.slot_check(arm, 'rhea')


def test_build_stages_union_identical_scopes_and_no_outer_gpu_lock(entry, tmp_path, monkeypatch, capsys):
    other = {**entry, 'name': 'glmf-min', 'family': 'glm5_flash', 'sparks': ['rhea', 'moa'], 'set': {'SPARK_COUNT': '2'}}
    arms = cards.arms_from(['base=b:s', 'candidate=c:s'])
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    cards.build_arms(['base=HEAD', 'candidate=HEAD'], arms, [entry, other], tmp_path, 'task', True)
    for arm in ('base', 'candidate'):
        cfg = cards.config(tmp_path / ('build-' + arm + '.config'))
        hosts = {cfg[f'SPARK_{i}_HOST'] for i in range(int(cfg['SPARK_COUNT']))}
        assert hosts == {'moa', 'rhea', 'ostrich', 'dodo', 'emu', 'kiwi'}
        assert cfg['SPARK_0_HOST'] == 'moa'
    output = capsys.readouterr().out
    assert output.count('identical build scopes') == 2
    assert 'build.lock' in output and 'gpu1.lock' not in output
    scopes = cards.build_scopes([entry])
    assert scopes['CUTEAFD_WIP_SPARK_TP_ROLES'] == 'tp3'
    assert scopes['CUTEAFD_WIP_EXPORT_LOCKS'] == 'on'
    # Qwen has no Spark FP8 layout: its scope is the NVFP4 family (CMake adds the coordinator FP8 sibling).
    qwen = {**entry, 'name': 'qwen38-nvfp4-min', 'family': 'qwen4', 'sparks': [], 'set': {'SPARK_COUNT': '0'}}
    families = cards.build_scopes([qwen, other])['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')
    assert 'qwen4:nvfp4' in families and 'qwen4:fp8' not in families and 'glmf:fp8' in families


def test_seed_host_moves_the_build_seed_off_moa(entry, tmp_path, monkeypatch):
    arms = cards.arms_from(['base=b:s'])
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    cards.build_arms(['base=HEAD'], arms, [entry], tmp_path, 'task', True, 'rhea')
    cfg = cards.config(tmp_path / 'build-base.config')
    assert cfg['SPARK_0_HOST'] == 'rhea'
    hosts = [cfg[f'SPARK_{i}_HOST'] for i in range(int(cfg['SPARK_COUNT']))]
    assert hosts.count('rhea') == 1 and set(entry['sparks']) <= set(hosts)


def test_legacy_build_lock_is_coarse_and_recorded(entry, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'legacy-full-build')
    arms = cards.arms_from(['base=b:s'])
    cards.build_arms(['base=old'], arms, [entry], tmp_path, 'task', True)
    text = capsys.readouterr().out
    assert 'legacy arm: GPU lock held for the full build' in text
    assert 'gpu1.lock' in text and 'flock -w 1800' in text
    assert arms['base']['build_lock_mode'] == 'legacy-full-build'
    assert 'source-base' in text and 'cp ' not in text


def test_source_lock_capability_detection(monkeypatch):
    texts = iter(['CUTEAFD_WIP_EXPORT_LOCKS', 'CUTEAFD_WIP_EXPORT_LOCK_FILES', 'old', 'old'])
    monkeypatch.setattr(cards, 'run', lambda cmd: SimpleNamespace(stdout=next(texts)))
    assert cards.build_lock_mode('new') == 'native-phase'
    assert cards.build_lock_mode('old') == 'legacy-full-build'


def test_requested_panels_preflight_and_report(entry, tmp_path):
    bad = {**entry, 'profile': 'panels:decode,fidelity'}
    with pytest.raises(ValueError, match='unknown requested panel: decode'):
        make(bad, tmp_path)
    job = make({**entry, 'profile': 'panels:decode_content,fidelity'}, tmp_path)
    report_dir = Path(job['state']) / 'reports' / 'one'
    report_dir.mkdir(parents=True)
    (report_dir / 'report.json').write_text(json.dumps({'panels': [{'id': 'fidelity'}]}))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'failed'
    assert 'decode_content' in row['error']
    (report_dir / 'report.json').write_text(json.dumps({'panels': [{'id': 'fidelity'}, {'id': 'decode_content'}]}))
    assert cards.summarize_job(job, 0)['status'] == 'pass'


def test_instance_hash_within_run_limit(entry, tmp_path):
    job = make({**entry, 'name': 'very-long-card-name-' * 5}, tmp_path, instance='long-' + 'a' * 35)
    instance = job['entry']['set']['INSTANCE']
    assert len(instance) == 41
    assert cards.NAME.fullmatch(instance)
    assert instance == cards.short_instance('long-' + 'a' * 35 + '-' + 'very-long-card-name-' * 5 + '-r1')
    assert cards.short_instance('a' * 50) != cards.short_instance('a' * 49 + 'b')


def test_matched_seed_same_across_arms_new_per_pair(entry, tmp_path, monkeypatch):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry])))
    state = tmp_path / 'state'
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--interleave', '--matched-prompts', '--arm', 'base=b:s', '--arm', 'candidate=c:s', '--state', str(state), '--dry-run'])
    assert cards.main() == 0
    jobs = json.loads((state / 'plan.json').read_text())['jobs']
    assert jobs[0]['nonce_seed'] == jobs[1]['nonce_seed']
    assert jobs[0]['nonce_seed'] != jobs[2]['nonce_seed']
    assert all('observer' in job['entry'] for job in jobs)


def test_matched_round_metrics_and_pair_hash_failure(tmp_path):
    events = [{'e': 'admit', 'id': 1, 'prompt': 42, 'max': 320}, {'e': 'round', 't0': 0, 't1': 10, 'req': [[1, 4, 4, 2, 3, 0, 0]]}, {'e': 'round', 't0': 10, 't1': 30, 'req': [[1, 4, 4, 2, 3, 0, 1]]}]
    (tmp_path / 'console-frames.jsonl').write_text(json.dumps({'ev': events}) + '\n')
    request = cards.matched_requests(tmp_path)[0]
    assert request['round_service_tok_s'] == 200
    assert request['emitted_tok_s'] is None  # Missing first/retire evidence is not invented.
    assert request['tokens_per_step'] == 3
    assert request['ms_per_step'] == 15
    rows = []
    for arm, token_hash in [('base', 'a'), ('candidate', 'b')]:
        state = tmp_path / arm
        state.mkdir()
        rows.append(dict(card='c', arm=arm, repeat=1, nonce_seed='seed', first_prompt_token_hash=token_hash, state=str(state), status='pass'))
    cards.validate_matched_pairs(rows, ['base', 'candidate'])
    assert all(row['status'] == 'failed' for row in rows)
    assert all((Path(row['state']) / 'exit').read_text() == '81\n' for row in rows)


def test_cleanup_foreign_root_refused(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    root = home / '.cache/cuteafd/builds/wip-test'
    root.mkdir(parents=True)
    (root / 'foreign').write_text('keep')
    script = cards.cleanup_script(dict(instance='test', slot='s'), 'task')
    result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'HOME': str(home)}, text=True, capture_output=True)
    assert result.returncode and 'refusing unowned' in result.stderr
    assert (root / 'foreign').exists()


def test_cleanup_foreign_container_refused(tmp_path):
    home = tmp_path / 'home'
    root = home / '.cache/cuteafd/builds/wip-test'
    root.mkdir(parents=True)
    (root / '.wip-card-owner.json').write_text(json.dumps(dict(task='task', instance='test')))
    script = cards.cleanup_script(dict(instance='test', slot='s'), 'task')
    prefix = '''import subprocess
subprocess.check_output=lambda cmd,**kw: '[{"Config":{"Env":["WIP_INSTANCE=foreign"]},"Mounts":[]}]' if 'inspect' in cmd else 'foreign-id'
subprocess.run=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('must not remove'))
'''
    result = subprocess.run([sys.executable, '-c', prefix + script], env={**os.environ, 'HOME': str(home)}, text=True, capture_output=True)
    assert result.returncode and 'refusing foreign container' in result.stderr
    assert root.exists()


def test_summary_medians_paired_not_ratio_of_medians():
    rows = [dict(card='c', arm=arm, repeat=i, status='pass', simulated=False, C1=v) for i, values in enumerate(((10, 20), (100, 110), (20, 40)), 1) for arm, v in zip(('base', 'candidate'), values)]
    medians, paired = cards.summary_math(rows, ['base', 'candidate'])
    assert medians[0]['C1'] == 20
    assert medians[1]['C1'] == 40
    delta = next(p for p in paired if p['metric'] == 'C1')
    assert delta['median_delta'] == 10
    assert delta['median_percent'] == 100
    assert delta['pairs'] == 3
    rows[0]['simulated'] = True
    assert next(p for p in cards.summary_math(rows, ['base', 'candidate'])[1] if p['metric'] == 'C1')['pairs'] == 2


def test_pool_precheck_and_red_square():
    assert cards.admitted_pool('\x1b[3madmitted_pool_tokens\x1b[0m=1048576') == 1048576
    png = cards.base64.b64decode(cards.red_square().split(',')[1])
    assert cards.struct.unpack('>II', png[16:24]) == (224, 224)


def test_native_export_lock_only_after_cargo(tmp_path):
    text = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    start = text.index('export_lock_fds=()')
    end = text.index("printf '%s'", start)
    phase = text[start:end]
    phase = phase[:phase.index('cmake \\\n')] + 'printf "export\\n" >> "$TRACE"\n' + phase[phase.index('for export_lock_fd in'):]
    tools = tmp_path / 'tools'
    tools.mkdir()
    mock = tools / 'flock'
    mock.write_text('#!/bin/sh\nprintf "flock %s\\n" "$*" >> "$TRACE"\n')
    mock.chmod(0o755)
    trace = tmp_path / 'trace'
    env = {**os.environ, 'PATH': str(tools) + ':' + os.environ['PATH'], 'TRACE': str(trace), 'CUTEAFD_WIP_EXPORT_LOCK_FILES': str(tmp_path / 'gpu1.lock')}
    subprocess.run(['bash', '-c', phase], env={**env, 'CUTEAFD_WIP_EXPORT_LOCKS': 'on'}, check=True)
    assert trace.read_text().splitlines()[0].startswith('flock -w 1800')
    assert trace.read_text().splitlines()[1] == 'export'
    assert trace.read_text().splitlines()[2].startswith('flock -u')
    trace.unlink()
    subprocess.run(['bash', '-c', phase], env={**env, 'CUTEAFD_WIP_EXPORT_LOCKS': 'off'}, check=True)
    assert trace.read_text() == 'export\n'
    assert text.index('cargo build') < start


@pytest.mark.parametrize('matched', [False, True])
def test_explicit_nonce_overrides_auto_and_inherited(entry, tmp_path, monkeypatch, capsys, matched):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry])))
    state = tmp_path / 'state'
    args = ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--arm', 'base=b:s',
            '--state', str(state), '--nonce-seed', 'explicit seed', '--dry-run']
    args += ['--interleave', '--matched-prompts', '--arm', 'candidate=c:s'] if matched else ['--matrix']
    monkeypatch.setenv('CUTEAFD_BENCH_NONCE_SEED', 'inherited')
    monkeypatch.setattr(sys, 'argv', args)
    assert cards.main() == 0
    jobs = json.loads((state / 'plan.json').read_text())['jobs']
    assert all(job['nonce_seed'] == 'explicit seed' for job in jobs)
    assert all('observer' in job['entry'] and 'console' in job['probes'] for job in jobs)
    for job in jobs:
        dest = Path(job['state'])
        report = dest / 'reports' / 'one' / 'report.json'
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({'baseline': {'quality': {'checks': [], 'status': 'pass'}}}))
        logs = dest / 'smoke' / 'logs'
        logs.mkdir(parents=True)
        (logs / 'one.coordinator.log').write_text('prompt_token_hash=0123456789abcdef')
        (dest / 'console-frames.jsonl').write_text(json.dumps({'ev': [{'e': 'round', 't0': 0, 't1': 10, 'req': [[1, 4, 4, 2, 3, 0, 0]]}]}) + '\n')
        row = cards.summarize_job(job, 0)
        assert row['status'] == 'pass' and row['first_prompt_token_hash'] == '0123456789abcdef'
        assert row['requests'][0]['steps'] == 1
    output = capsys.readouterr().out
    assert "CUTEAFD_BENCH_NONCE_SEED='explicit seed'" in output
    assert 'inherited' not in output
    assert all(json.loads((Path(job['state']) / 'job.json').read_text())['nonce_seed'] == 'explicit seed' for job in jobs)


def test_arm_overrides_only_selected_arm_and_card(entry, tmp_path, monkeypatch, capsys):
    kit = tmp_path / 'kit'
    kit.mkdir()
    other = {**entry, 'name': 'other'}
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry, other])))
    state = tmp_path / 'state'
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', 'other',
        '--interleave', '--repeats', '1', '--arm', 'off=b:s', '--arm', 'on=c:s', '--state', str(state),
        '--set', 'v41-min:GLM5_FLASH_PREFILL_BATCH=off', '--set', 'v41-min:GLM5_FLASH_VERIFY_POLICY=cost',
        '--arm-set', 'on:v41-min:GLM5_FLASH_PREFILL_BATCH=on',
        '--arm-set', 'on:v41-min:GLM5_FLASH_VERIFY_POLICY=chain', '--dry-run'])
    assert cards.main() == 0
    jobs = json.loads((state / 'plan.json').read_text())['jobs']
    for job in jobs:
        settings = job['entry']['set']
        base = cards.config(Path(job['state']) / 'base.config')
        if job['card'] == 'other':
            assert 'GLM5_FLASH_PREFILL_BATCH' not in settings
        else:
            enabled = job['arm'] == 'on'
            assert settings['GLM5_FLASH_PREFILL_BATCH'] == ('on' if enabled else 'off')
            assert settings['GLM5_FLASH_VERIFY_POLICY'] == ('chain' if enabled else 'cost')
            assert base['GLM5_FLASH_VERIFY_POLICY'] == settings['GLM5_FLASH_VERIFY_POLICY']
    assert '"GLM5_FLASH_VERIFY_POLICY": "chain"' in capsys.readouterr().out


@pytest.mark.parametrize('override,match', [('foreign:v41-min:MODEL_ID=x', 'unknown override arm'),
    ('base:foreign:MODEL_ID=x', 'selected cards'), ('base:v41-min:API_KEY_FILE=x', 'driver owns')])
def test_arm_override_validation(entry, tmp_path, monkeypatch, override, match):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry])))
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix',
        '--arm', 'base=b:s', '--state', str(tmp_path / 'state'), '--arm-set', override, '--dry-run'])
    with pytest.raises(ValueError, match=match):
        cards.main()


@pytest.mark.parametrize('seed', [None, 'explicit'])
def test_runtime_seed_is_explicit_only(entry, tmp_path, monkeypatch, seed):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry])))
    state = tmp_path / 'state'
    args = ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix', '--arm', 'base=b:s', '--state', str(state)]
    if seed is not None:
        args += ['--nonce-seed', seed]
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setenv('CUTEAFD_BENCH_NONCE_SEED', 'inherited')
    monkeypatch.setattr(cards, 'assert_absent', lambda job: None)
    monkeypatch.setattr(cards, 'slot_check', lambda arm, host: json.dumps(dict(slot=arm['slot'], seal_sha256='seal', artifact_manifest_sha256='artifacts')))
    seen = []
    def detached(cmd, log, exit_file, env):
        seen.append(env)
        exit_file.write_text('0\n')
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(cards, 'detached', detached)
    monkeypatch.setattr(cards, 'summarize_job', lambda job, code: dict(card=job['card'], arm=job['arm'], repeat=1, simulated=True, status='pass'))
    monkeypatch.setattr(cards, 'write_summary', lambda *args: None)
    assert cards.main() == 0
    assert seen[0].get('CUTEAFD_BENCH_NONCE_SEED') == seed


@pytest.mark.parametrize('status', ['pass', 'failed'])
def test_summary_extracts_exact_cache_check(entry, tmp_path, status):
    job = make(entry, tmp_path)
    report = Path(job['state']) / 'reports' / 'one' / 'report.json'
    report.parent.mkdir(parents=True)
    check = {'id': 'cache_exact', 'status': status, 'summary': 'byte-exact restore'}
    report.write_text(json.dumps({'baseline': {'quality': {'checks': [check], 'status': status}}}))
    row = cards.summarize_job(job, 0)
    assert row['cache'] == check
    assert row['status'] == ('failed' if status == 'failed' else 'pass')


def test_shared_wip_same_slot_and_distinct_serving_instances(entry, tmp_path):
    arms = cards.arms_from(['off=usage-off:s', 'on=usage-on:s'])
    cards.shared_wip(arms,['off=usage-build','on=usage-build'],[],1)
    jobs = [cards.generate(entry,n,a,tmp_path,tmp_path/'api-key',{},set(),None,1) for n,a in arms.items()]
    assert {j['entry']['set']['WIP_INSTANCE'] for j in jobs} == {'usage-build'}
    assert len({j['entry']['set']['INSTANCE'] for j in jobs}) == 2
    arms['on']['slot'] = 'different'
    with pytest.raises(ValueError,match='identical slots'):
        cards.shared_wip(arms,[],[],1)
    for builds,parallel in [(['off=HEAD'],1),([],2)]:
        with pytest.raises(ValueError,match='refuses'):
            cards.shared_wip(cards.arms_from(['off=off:s']),['off=shared'],builds,parallel)


def test_shared_external_wip_cleanup_never_deletes_build(monkeypatch):
    monkeypatch.setattr(cards,'cleanup_script',lambda *a: pytest.fail('external build cleanup'))
    cards.cleanup({'off':{'instance':'off','slot':'s','wip_instance':'shared'},'on':{'instance':'on','slot':'s','wip_instance':'shared'}},'task',False)


@pytest.mark.parametrize('dry', [True, False])
def test_release_arm_uses_same_launcher_without_wip(entry, tmp_path, monkeypatch, dry):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry])))
    state = tmp_path / 'state'
    argv = ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--interleave',
            '--repeats', '1', '--arm', 'candidate=v3s1a-c:s',
            '--arm-release', 'release=v3s1a-release:v2.0.0', '--state', str(state),
            '--repo', '/same-candidate-launcher']
    if dry:
        argv.append('--dry-run')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(cards, 'assert_absent', lambda job: None)
    monkeypatch.setattr(cards, 'slot_check', lambda arm, host: json.dumps(
        dict(image='release', image_id='sha256:abc', repo_digests=['digest']) if 'release' in arm
        else dict(slot='s', seal_sha256='seal', artifact_manifest_sha256='artifacts')))
    commands = []
    def detached(cmd, log, exit_file, env):
        commands.append(cmd)
        exit_file.write_text('0\n')
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(cards, 'detached', detached)
    monkeypatch.setattr(cards, 'summarize_job', lambda job, code: dict(card=job['card'], arm=job['arm'], repeat=1, simulated=True, status='pass'))
    monkeypatch.setattr(cards, 'write_summary', lambda *args: None)
    assert cards.main() == 0
    jobs = json.loads((state / 'plan.json').read_text())['jobs']
    release = next(job for job in jobs if job['arm'] == 'release')
    assert release['entry']['run_args'] == []
    values = release['entry']['set']
    assert 'WIP_INSTANCE' not in values and 'WIP_ROOT' not in values
    assert values['COORDINATOR_DOCKER_INFERENCE'] == 'ghcr.io/tpurtell/cuteafd-coordinator:v2.0.0'
    assert values['SPARK_EXPERT_DOCKER_INFERENCE'] == 'ghcr.io/tpurtell/cuteafd-spark-expert:v2.0.0'
    assert jobs[0]['entry']['run_args'] == ['--wip', 's']
    assert all(cmd[cmd.index('--repo') + 1] == '/same-candidate-launcher' for cmd in commands)
    if not dry:
        saved = json.loads((Path(release['state']) / 'job.json').read_text())
        assert saved['artifacts']['raptor']['image_id'] == 'sha256:abc'


def test_release_arm_validation_and_cleanup(entry, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='unique'):
        cards.arms_from(['base=b:s'], ['base=r:v2.0.0'])
    with pytest.raises(ValueError, match='unique'):
        cards.arms_from(['base=b:s'], ['release=b:v2.0.0'])
    with pytest.raises(ValueError, match='invalid release'):
        cards.arms_from([], ['release=r:v2.0.0/other'])
    arms = cards.arms_from([], ['release=r:v2.0.0'])
    with pytest.raises(ValueError, match='invalid --arm-wip'):
        cards.shared_wip(arms, ['release=shared'], [], 1)
    with pytest.raises(ValueError, match='refuses release'):
        cards.build_arms(['release=HEAD'], arms, [entry], tmp_path, 'task', True)
    monkeypatch.setattr(cards, 'cleanup_script', lambda *a: pytest.fail('release cleanup'))
    cards.cleanup(arms, 'task', False)


def test_release_arm_checks_local_and_remote_images(monkeypatch):
    calls = []
    monkeypatch.setattr(cards, 'run', lambda cmd: calls.append(cmd) or SimpleNamespace(
        stdout=json.dumps([dict(Id='sha256:abc', RepoDigests=['digest'])])))
    arm = cards.arms_from([], ['release=r:v2.0.0'])['release']
    for host in ('raptor', 'kiwi'):
        meta = json.loads(cards.slot_check(arm, host))
        assert meta['image_id'] == 'sha256:abc'
    assert calls[0] == ['docker', 'image', 'inspect', 'ghcr.io/tpurtell/cuteafd-coordinator:v2.0.0']
    assert calls[1][:4] == ['ssh', '-o', 'BatchMode=yes', 'kiwi']
    assert 'cuteafd-spark-expert:v2.0.0' in calls[1][-1]
