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


def test_console_observer_records_policy_stats(entry, tmp_path, monkeypatch):
    job = cards.generate(entry, 'candidate', dict(instance='policy-test', slot='s'),
                         tmp_path, tmp_path / 'key', {}, {'console'}, None, 1)
    stop = tmp_path / 'observer.stop'
    monkeypatch.setenv('CUTEAFD_SMOKE_STOP', str(stop))
    monkeypatch.setenv('CUTEAFD_SMOKE_URL', 'http://localhost:8500')
    calls = []

    def http(url, path):
        calls.append(path)
        if path == '/v1/stats':
            stop.touch()
            return {'dspark_policy': {'solo': {'concurrency_buckets': [{'engaged': True}]}}}
        return {'type': 'snapshot'}

    monkeypatch.setattr(cards, 'http', http)
    monkeypatch.setattr(cards, 'console_frames', lambda *args: None)
    monkeypatch.setattr(cards.time, 'sleep', lambda _: None)
    cards.observe(job)
    record = json.loads((Path(job['state']) / 'policy-stats.jsonl').read_text())
    assert record['stats']['dspark_policy']['solo']['concurrency_buckets'][0]['engaged']
    assert calls == ['/v1/console/snapshot', '/v1/stats']


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


@pytest.mark.parametrize('source', ['config', 'set', 'override'])
def test_glm_ring_expectation_is_observer_only(entry, tmp_path, source):
    entry['observer'] = dict(argv=['python3', 'ring-observer.py'], timeout_s=300)
    overrides = {}
    if source == 'config':
        with Path(entry['config']).open('a') as stream:
            stream.write('BENCH_EXPECT_GLM_RING=on\n')
    elif source == 'set':
        entry['set']['BENCH_EXPECT_GLM_RING'] = 'on'
    else:
        overrides['BENCH_EXPECT_GLM_RING'] = 'on'
    job = cards.generate(entry, 'candidate', dict(instance='ring-test', slot='s'),
                         tmp_path, tmp_path / 'key', overrides, set(), None, 1)
    generated = json.loads((Path(job['state']) / 'matrix.json').read_text())['entries'][0]
    assert 'BENCH_EXPECT_GLM_RING' not in generated['set']
    assert 'BENCH_EXPECT_GLM_RING' not in Path(generated['config']).read_text()
    assert generated['observer']['argv'] == ['env', 'BENCH_EXPECT_GLM_RING=on', 'python3', 'ring-observer.py']
    assert entry['observer']['argv'] == ['python3', 'ring-observer.py']


def test_glm_ring_expectation_requires_preserved_observer(entry, tmp_path):
    entry['set']['BENCH_EXPECT_GLM_RING'] = 'on'
    with pytest.raises(ValueError, match='requires an observer'):
        make(entry, tmp_path)
    entry['observer'] = dict(argv=['python3', 'ring-observer.py'])
    with pytest.raises(ValueError, match='probe replacement'):
        cards.generate(entry, 'candidate', dict(instance='ring-test', slot='s'),
                       tmp_path, tmp_path / 'key', {}, {'memory'}, None, 1)


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


@pytest.mark.parametrize('dry', [False, True])
def test_build_only_builds_without_generating_or_launching_jobs(tmp_path, entry, monkeypatch, dry):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=[entry])))
    state = tmp_path / 'state'
    calls = []

    def build(items, arms, selected, dest, task, dry_run, seed):
        calls.append((items, selected, dest, task, dry_run, seed))
        arms['base']['root'] = '/sealed/root'

    def forbidden(*args, **kwargs):
        pytest.fail('build-only must not generate or launch smoke jobs')

    monkeypatch.setattr(cards, 'build_arms', build)
    monkeypatch.setattr(cards, 'generate', forbidden)
    monkeypatch.setattr(cards, 'detached', forbidden)
    argv = ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix',
            '--arm', 'base=test:s', '--state', str(state), '--task', 'build-only',
            '--build', 'base=HEAD', '--build-only', '--seed-host', 'rhea',
            '--set', 'v41-min:SPARK_COUNT=4']
    monkeypatch.setattr(sys, 'argv', argv + (['--dry-run'] if dry else []))
    assert cards.main() == 0
    assert len(calls) == 1
    items, selected, dest, task, dry_run, seed = calls[0]
    assert (items, dest, task, dry_run, seed) == (['base=HEAD'], state, 'build-only', dry, 'rhea')
    assert selected[0]['set']['SPARK_COUNT'] == '4'
    assert len(selected[0]['sparks']) == 4
    plan = json.loads((state / 'plan.json').read_text())
    assert plan['jobs'] == [] and plan['arms']['base']['root'] == '/sealed/root'
    assert not (state / 'api-key').exists()


@pytest.mark.parametrize('extra', [[], ['--build', 'base=HEAD', '--cleanup']])
def test_build_only_requires_build_and_refuses_cleanup(tmp_path, entry, monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--arm', 'base=test:s', '--build-only', *extra])
    with pytest.raises(SystemExit) as error:
        cards.main()
    assert error.value.code == 2


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


@pytest.mark.parametrize('family,tag,count', [('glm5', 'glm', 4), ('glm5_flash', 'glmf', 2), ('qwen4', 'qwen4', 4)])
@pytest.mark.parametrize('declared_by', ['name', 'model', 'format'])
def test_matrix_nvfp4_required_spark_packages_are_in_build_set(entry, family, tag, count, declared_by):
    values = {'SPARK_COUNT': str(count)}
    name = 'unlabelled-native-card'
    if declared_by == 'name':
        name = tag + '-nvfp4-min'
    elif declared_by == 'model':
        values['MODEL_ID'] = 'nvidia/Model-NVFP4'
    else:
        values['EXPERT_FORMAT'] = 'nvfp4'
    matrix_entry = {**entry, 'name': name, 'family': family, 'set': values}
    build = cards.build_scopes([matrix_entry])
    built = set(build['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';'))
    assert f'{tag}={count}' in build['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'].split(';')
    # native/cmake/shared/fp8_moe.cmake expands nvfp4 to both historical
    # nvfp4 and mandatory all-row nvfp4a4 packages at supported TP layouts.
    cmake = (ROOT / 'native/cmake/shared/fp8_moe.cmake').read_text()
    assert tag + ':nvfp4' in built
    assert 'if(entry MATCHES ":nvfp4$")' in cmake
    assert 'list(APPEND CUTEAFD_FP8_MOE_ENTRIES "${entry}a4")' in cmake
    assert 'list(APPEND requested_layouts "tp${count}")' in cmake
    assert count in (2, 4)
    assert 'glmfdense:nvfp4' not in built


def test_build_stages_union_identical_scopes_and_no_outer_gpu_lock(entry, tmp_path, monkeypatch, capsys):
    other = {**entry, 'name': 'glmf-min', 'family': 'glm5_flash', 'sparks': ['rhea', 'moa'], 'set': {'SPARK_COUNT': '2'}}
    arms = cards.arms_from(['base=b:s', 'candidate=c:s'])
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    # A revision whose wip.sh predates build slots keeps the outer build.lock.
    monkeypatch.setattr(cards, 'build_slot_capable', lambda rev: False)
    monkeypatch.setattr(cards, 'seed_option_capable', lambda rev: True)
    cards.build_arms(['base=HEAD', 'candidate=HEAD'], arms, [entry, other], tmp_path, 'task', True)
    for arm in ('base', 'candidate'):
        cfg = cards.config(tmp_path / ('build-' + arm + '.config'))
        hosts = [cfg[f'SPARK_{i}_HOST'] for i in range(int(cfg['SPARK_COUNT']))]
        # The union's five serving hosts, padded to six ranks with a pool Spark; the
        # default seed (build-only kakapo) is wip.sh's WIP_SEED_HOST, never a rank (#117).
        assert set(hosts) == {'moa', 'rhea', 'ostrich', 'dodo', 'emu', 'kiwi'} and len(hosts) == 6
        assert cards.DEFAULT_SEED not in hosts
    output = capsys.readouterr().out
    assert output.count('identical build scopes') == 2
    assert 'build.lock' in output and 'gpu1.lock' not in output
    scopes = cards.build_scopes([entry])
    assert scopes['CUTEAFD_WIP_SPARK_TP_ROLES'] == 'tp3'
    assert scopes['CUTEAFD_WIP_EXPORT_LOCKS'] == 'on'
    # Qwen has no Spark FP8 layout: its scope is the NVFP4 family (CMake adds the coordinator FP8 sibling).
    qwen = {**entry, 'name': 'qwen38-nvfp4-min', 'family': 'qwen4', 'sparks': [], 'set': {'SPARK_COUNT': '0'}}
    built = cards.build_scopes([qwen, other])
    families = built['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')
    # A 0-Spark card adds coordinator scopes only (per-card derivation).
    assert 'qwen4:nvfp4' not in families and 'glmf:fp8' in families
    coordinator = built['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert 'qwen4:nvfp4' in coordinator and 'qwen4:fp8' not in coordinator
    nvfp4 = [{**entry, 'name': 'glm53-nvfp4-min', 'family': 'glm5'},
             {**other, 'name': 'glmf-nvfp4-min', 'family': 'glm5_flash'}]
    scopes = cards.build_scopes(nvfp4)
    families = set(scopes['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';'))
    coordinator = set(scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';'))
    assert {'glm:nvfp4', 'glmf:nvfp4'} <= families
    assert 'glmfdense:nvfp4' not in families
    assert coordinator == families | {'glmfdense:nvfp4'}
    assert 'glm:fp8' not in families and 'glmf:fp8' not in families
    wip = (cards.REPO / 'wip.sh').read_text()
    coordinator_leg = wip.split('build_coordinator() {', 1)[1].split('build_expert() (', 1)[0]
    assert 'CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES:-${CUTEAFD_WIP_EXPERT_FAMILIES:-}' in coordinator_leg
    expert_leg = wip.split('build_expert() (', 1)[1]
    assert 'CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES' not in expert_leg


@pytest.mark.parametrize('name,tag', [('v4-flash-min', 'dsv4f'), ('v4-pro-exl3-max', 'dsv4p')])
@pytest.mark.parametrize('gpus,override,wants_tp2', [([0], None, False), ([0, 1], None, True), ([0], '2', True), ([0, 1], '1', False)])
def test_v4_scopes_keep_full_experts_and_add_dual_rtx_halves(entry, name, tag, gpus, override, wants_tp2):
    values = {'SPARK_COUNT': '4'}
    if override is not None:
        values['RTX_GPUS'] = override
    scope = cards.build_scopes([{**entry, 'name': name, 'family': 'deepseek_v4', 'gpus': gpus, 'set': values}])
    families = set(scope['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';'))
    assert tag + ':rtx_backbone' in families
    assert (tag + ':rtx_tp2' in families) == wants_tp2
    assert tag + ':spark' in families


def test_seed_host_moves_the_build_seed_off_moa(entry, tmp_path, monkeypatch):
    arms = cards.arms_from(['base=b:s'])
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    for separate in (False, True):
        monkeypatch.setattr(cards, 'seed_option_capable', lambda rev: separate)
        cards.build_arms(['base=HEAD'], arms, [entry], tmp_path, 'task', True, 'rhea')
        cfg = cards.config(tmp_path / 'build-base.config')
        hosts = [cfg[f'SPARK_{i}_HOST'] for i in range(int(cfg['SPARK_COUNT']))]
        # Legacy wip.sh builds on SPARK_0_HOST; a separate seed is passed as WIP_SEED_HOST.
        assert (hosts[0] == 'rhea') == (not separate)
        assert set(entry['sparks']) <= set(hosts) and hosts.count('rhea') == (0 if separate else 1)


def test_legacy_build_lock_is_coarse_and_recorded(entry, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'legacy-full-build')
    monkeypatch.setattr(cards, 'toolchain_hash', lambda rev: '0123456789abcdef')
    arms = cards.arms_from(['base=b:s'])
    cards.build_arms(['base=old'], arms, [entry], tmp_path, 'task', True)
    text = capsys.readouterr().out
    assert 'legacy arm: GPU lock held for the full build' in text
    assert 'hwlock.py --locks gpu1 --wait 1800' in text
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


def test_matched_prefill_only_sweep_needs_no_decode_rounds(tmp_path):
    # #72: a prefill-only sweep (max_tokens 1) emits no decode rounds; its pairs
    # match by prompt token hash alone.
    def job(entry):
        state = tmp_path / entry['name']
        (state / 'smoke' / 'logs').mkdir(parents=True)
        (state / 'smoke' / 'logs' / 'x.coordinator.log').write_text('prompt_token_hash="73b4ebf8f4958775"\n')
        (state / 'reports' / entry['name']).mkdir(parents=True)
        (state / 'reports' / entry['name'] / 'sweep.json').write_text(json.dumps({'points': [], 'prefill': []}))
        return dict(card='c', arm='a', repeat=1, simulated=False, state=str(state), nonce_seed='s', entry=entry)
    prefill = cards.summarize_job(job(dict(name='p', profile='panels:', prefill_lengths=[8192], set={}, sparks=[])), 0)
    assert prefill['status'] == 'pass' and prefill['first_prompt_token_hash'] == '73b4ebf8f4958775'
    decode = cards.summarize_job(job(dict(name='d', profile='panels:', levels=[1, 16], set={}, sparks=[])), 0)
    assert decode['status'] == 'failed' and 'per-request console rounds' in decode['error']


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


def test_cleanup_leaves_another_instances_legacy_slot_registration(tmp_path):
    # The legacy wip-slots/<slot>.json is shared by every instance using that
    # slot name; cleaning one instance must neither remove nor refuse on it.
    home = tmp_path / 'home'
    root = home / '.cache/cuteafd/builds/wip-test'
    root.mkdir(parents=True)
    (root / '.wip-card-owner.json').write_text(json.dumps(dict(task='task', instance='test')))
    slots = home / '.cache/cuteafd/wip-slots'
    slots.mkdir(parents=True)
    (slots / 's.json').write_text(json.dumps(dict(wip_instance='other', slot='s')))
    (slots / 'test').mkdir()
    (slots / 'test' / 's.json').write_text(json.dumps(dict(wip_instance='test', slot='s')))
    script = cards.cleanup_script(dict(instance='test', slot='s'), 'task')
    prefix = 'import subprocess\nsubprocess.check_output=lambda cmd,**kw: ""\n'
    result = subprocess.run([sys.executable, '-c', prefix + script], env={**os.environ, 'HOME': str(home)}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (slots / 's.json').exists() and not (slots / 'test' / 's.json').exists() and not root.exists()
    # A foreign entry in the instance's own registration is still refused.
    root.mkdir(parents=True)
    (root / '.wip-card-owner.json').write_text(json.dumps(dict(task='task', instance='test')))
    (slots / 'test' / 's.json').write_text(json.dumps(dict(wip_instance='other', slot='s')))
    result = subprocess.run([sys.executable, '-c', prefix + script], env={**os.environ, 'HOME': str(home)}, text=True, capture_output=True)
    assert result.returncode and 'refusing foreign registration' in result.stderr


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


def _wip_function(name, end):
    text = (ROOT / 'wip.sh').read_text()
    start = text.index(name)
    helpers = text[text.index('# Native/AOT outputs are cached by input key'):text.index('build_coordinator() {')]
    return text[text.index('wip_hwlock() {'):text.index('\n}\n', text.index('wip_hwlock() {')) + 3] + \
        helpers + text[start:text.index(end, start)]


def _trace_tools(tmp_path, extra=None):
    tools = tmp_path / 'tools'
    tools.mkdir()
    scripts = {
        'ssh': '#!/bin/sh\nprintf "ssh %s\\n" "$*" >> "$TRACE"\nprintf "image-id\\n"\n',
        'docker': '#!/bin/sh\nprintf "docker %s\\n" "$*" >> "$TRACE"\nprintf "image-id\\n"\n',
        'flock': '#!/bin/sh\nprintf "flock %s\\n" "$*" >> "$TRACE"\n',
        # hwlock stand-in: record the turn, then run the command it guards.
        'python3': '#!/bin/sh\nprintf "turn %s\\n" "$*" >> "$TRACE"\nwhile [ "$1" != -- ]; do shift; done; shift\nprintf "begin\\n" >> "$TRACE"\n"$@"\nrc=$?\nprintf "end\\n" >> "$TRACE"\nexit $rc\n',
        'timeout': '#!/bin/sh\nwhile [ "${1#-}" != "$1" ]; do shift; done; shift\nexec "$@"\n',
        **(extra or {})}
    for name, script in scripts.items():
        (tools / name).write_text(script)
        (tools / name).chmod(0o755)
    return tools


def test_spark_leg_holds_one_seed_turn_for_its_whole_duration(tmp_path):
    function = _wip_function('build_expert() (', '\nwip_build_coordinator=0') + '\nnative_store_has() { return 1; }\n'
    tools = _trace_tools(tmp_path)
    home = tmp_path / 'home'
    (home / '.cache/cuteafd').mkdir(parents=True)
    trace = tmp_path / 'trace'
    setup = f'''set -euo pipefail
repo_root={str(ROOT)!r}
sync_seed_source() {{ :; }}
release_die() {{ exit 2; }}
slot=test seed_host=rhea spark_container=owned SPARK_EXPERT_DOCKER_DEV=image
wip_spark_tp_roles='tp3;tp6' bf16_families= audio_aot=OFF
CUTEAFD_WIP_EXPORT_LOCKS=on wip_export_locks=on
'''
    subprocess.run(['bash', '-c', setup + function + '\nbuild_expert'], check=True,
                   env={**os.environ, 'HOME': str(home), 'PATH': str(tools) + ':' + os.environ['PATH'], 'TRACE': str(trace)})
    lines = trace.read_text().splitlines()
    rust = next(i for i, line in enumerate(lines) if line.endswith('/wip/output/expert rust'))
    turn = next(i for i, line in enumerate(lines) if line.startswith('turn '))
    native = next(i for i, line in enumerate(lines) if line.endswith('/wip/output/expert native') and not line.startswith('turn '))
    end = lines.index('end', turn)
    finalize = next(i for i, line in enumerate(lines) if 'finalize-wip-slot.sh' in line)
    assert 'scripts/hw/hwlock.py --locks rhea --wait 1800 --' in lines[turn]
    # One fair turn on the seed covers the whole Spark leg (Cargo, export,
    # finalize): Spark CPU work never overlaps cards serving from that host.
    assert turn < rust < native < finalize < end
    assert sum(line.startswith('turn ') for line in lines) == 1
    assert not any(line.startswith('flock') for line in lines)


def test_coordinator_export_runs_under_a_held_turn_and_rust_under_none(tmp_path):
    function = _wip_function('build_coordinator() {', '\nbuild_expert() (') + '\nnative_store_has() { return 1; }\n'
    tools = _trace_tools(tmp_path)
    trace = tmp_path / 'trace'
    setup = f'''set -euo pipefail
repo_root={str(ROOT)!r}
sync_local_source() {{ :; }}
release_die() {{ exit 2; }}
cuteafd_compiler_cache_warn() {{ :; }}
slot=test coordinator_container=owned COORDINATOR_DOCKER_DEV=image bf16_families= audio_aot=OFF
CUTEAFD_WIP_EXPORT_LOCKS=on wip_export_locks=on wip_export_gpu_lock=gpu1
# The fair turn as the only mode (off-queue modes: test_wip_coordinator_export_modes.py).
CUTEAFD_WIP_COORDINATOR_EXPORT=turn
'''
    modes = _wip_function('wip_coordinator_export_modes() {', '\n# wip_remote_export:')
    subprocess.run(['bash', '-c', setup + modes + function + '\nbuild_coordinator'], check=True,
                   env={**os.environ, 'PATH': str(tools) + ':' + os.environ['PATH'], 'TRACE': str(trace)})
    lines = trace.read_text().splitlines()
    rust = next(i for i, line in enumerate(lines) if line.endswith('/wip/output/coordinator rust'))
    turn = next(i for i, line in enumerate(lines) if line.startswith('turn '))
    native = next(i for i, line in enumerate(lines) if line.endswith('/wip/output/coordinator native') and not line.startswith('turn '))
    assert 'scripts/hw/hwlock.py --locks gpu1 --wait 1800 --' in lines[turn]
    assert rust < turn < native < lines.index('end', turn)
    assert all('CUTEAFD_WIP_EXPORT_LOCKS=off' in line for line in lines if 'build-wip-artifacts.sh' in line)
    wip = (ROOT / 'wip.sh').read_text()
    assert '/wip-export-gpu.lock' not in wip


def test_split_native_phase_requires_matching_rust_source_and_cache_mode():
    text = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    assert 'phase="${6:-all}"' in text
    assert '[[ "$phase" == native || "$phase" == export ]] || cuteafd_build_cache_cargo_offline' in text
    assert 'rust_phase_marker="$CARGO_TARGET_DIR/.wip-rust-phase-fingerprint"' in text
    assert '"$wip_rust_fingerprint" && -x "$CARGO_TARGET_DIR/release/cuteafd"' in text
    assert '[[ "$phase" != rust ]] || exit 0' in text


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


@pytest.mark.parametrize('status', ['pass', 'fail', 'failed'])
def test_summary_extracts_exact_cache_check(entry, tmp_path, status):
    job = make(entry, tmp_path)
    report = Path(job['state']) / 'reports' / 'one' / 'report.json'
    report.parent.mkdir(parents=True)
    check = {'id': 'cache_exact', 'status': status, 'summary': 'byte-exact restore'}
    report.write_text(json.dumps({'baseline': {'quality': {'checks': [check], 'status': status}}}))
    row = cards.summarize_job(job, 0)
    assert row['cache'] == check
    assert row['status'] == ('pass' if status == 'pass' else 'failed')


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


def _sealed_slot(root, slot='s', role='coordinator', instance='inst'):
    import hashlib
    p = root / 'slots' / slot / role
    (p / 'workspace/.cuteafd-wip').mkdir(parents=True)
    (p / 'SOURCE_SHA256SUMS').write_text('source\n')
    (p / 'workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS').write_text('artifacts\n')
    meta = json.dumps(dict(slot=slot, role=role, wip_instance=instance,
        source_manifest_sha256=hashlib.sha256(b'source\n').hexdigest(),
        artifact_manifest_sha256=hashlib.sha256(b'artifacts\n').hexdigest())).encode()
    (p / 'META.json').write_bytes(meta)
    (p / 'FINGERPRINT').write_text(hashlib.sha256(meta).hexdigest() + '\n')
    return p


def test_slot_check_reads_host_root_without_any_container(tmp_path, monkeypatch):
    # A build stops its WIP containers; the seal check reads the bind-mounted
    # root on the host and never execs, starts or needs a container.
    root = tmp_path / 'wip-inst'
    slot = _sealed_slot(root)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'docker.log'
    (bin_dir / 'docker').write_text(f'#!/bin/sh\necho "$@" >> {log}\nexit 1\n')
    (bin_dir / 'docker').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    arm = dict(instance='serving', wip_instance='inst', slot='s', root=str(root))
    meta = json.loads(cards.slot_check(arm, 'raptor'))
    assert meta['seal_source'] == 'host:' + str(slot)
    assert not log.exists()
    # Without a recorded root: the stopped container's /wip mount, inspected only.
    arm.pop('root')
    (bin_dir / 'docker').write_text(f'#!/bin/sh\necho "$@" >> {log}\n[ "$1" = inspect ] && echo {root}\n')
    meta = json.loads(cards.slot_check(arm, 'raptor'))
    assert meta['seal_source'] == 'host:' + str(slot)
    assert all(line.startswith('inspect ') for line in log.read_text().splitlines())
    # Integrity checks are unchanged: a tampered artifact manifest is refused.
    (slot / 'workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS').write_text('changed\n')
    with pytest.raises(ValueError, match='raptor: missing or unsealed WIP slot serving:s: .*ARTIFACT_SHA256SUMS seal mismatch'):
        cards.slot_check(arm, 'raptor')


def _kit(tmp_path, *entries):
    kit = tmp_path / 'kit'
    kit.mkdir(exist_ok=True)
    (kit / 'matrix-main.json').write_text(json.dumps(dict(entries=list(entries))))
    return kit


def test_panels_levels_and_post_load_reach_every_job(entry, tmp_path, monkeypatch):
    kit = _kit(tmp_path, entry)
    state = tmp_path / 'state'
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--interleave', '--repeats', '1',
        '--arm', 'a=b:s', '--arm', 'c=d:s', '--state', str(state), '--panels', 'none', '--levels', '1,16', '--post-load-c1', '--dry-run'])
    assert cards.main() == 0
    for job in json.loads((state / 'plan.json').read_text())['jobs']:
        matrix = json.loads((Path(job['state']) / 'matrix.json').read_text())['entries'][0]
        assert matrix['profile'] == 'panels:'
        assert matrix['levels'] == [1, 16] and matrix['post_load_c1'] is True


@pytest.mark.parametrize('extra,match', [(['--post-load-c1'], 'needs --levels'), (['--panels', 'none'], 'runs nothing'),
    (['--levels', '1,1'], 'distinct'), (['--panels', 'bogus'], 'unknown requested panel')])
def test_subset_flags_refuse_nonsense(entry, tmp_path, monkeypatch, extra, match):
    kit = _kit(tmp_path, entry)
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix',
        '--arm', 'a=b:s', '--state', str(tmp_path / 'state'), *extra, '--dry-run'])
    with pytest.raises((SystemExit, ValueError), match=None) as caught:
        cards.main()
    if caught.type is ValueError:
        assert match in str(caught.value)


def test_profile_dir_is_per_job_and_driver_owned(entry, tmp_path):
    job = cards.generate(entry, 'a', dict(instance='p', slot='s'), tmp_path, tmp_path / 'key',
                         {'PROFILE': 'expert-timing,transport-timing'}, set(), None, 1)
    values = job['entry']['set']
    assert values['PROFILE_DIR'] == str(Path(job['state']) / 'profile')
    assert (Path(job['state']) / 'profile').is_dir()
    assert 'PROFILE_DIR=' + values['PROFILE_DIR'] in (Path(job['state']) / 'base.config').read_text()
    with pytest.raises(ValueError, match='PROFILE_DIR is owned by the driver'):
        cards.generate(entry, 'a', dict(instance='p', slot='s'), tmp_path, tmp_path / 'key', {'PROFILE_DIR': '/x'}, set(), None, 1)


def test_profile_collection_pulls_each_spark_and_records_failures(entry, tmp_path, monkeypatch):
    job = cards.generate(entry, 'a', dict(instance='p', slot='s'), tmp_path, tmp_path / 'key', {'PROFILE': 'expert-timing'}, set(), None, 1)
    seen = []
    def fake(cmd, **kwargs):
        seen.append(cmd)
        if 'emu:' in cmd[-2]:
            raise subprocess.CalledProcessError(23, cmd, stderr='no such dir')
        Path(cmd[-1], 'expert-timing-' + cmd[-2].split(':')[0] + '-rank0.jsonl').write_text('{}\n')
    monkeypatch.setattr(cards.subprocess, 'run', fake)
    result = cards.collect_profile(job)
    assert [c[-2].split(':')[0] for c in seen] == ['ostrich', 'dodo', 'emu']
    assert result['hosts']['emu'].startswith('failed') and result['hosts']['ostrich'] == 'collected'
    assert 'expert-timing-dodo-rank0.jsonl' in result['files']


def test_nsys_profile_frames_the_sweep_with_a_window(entry, tmp_path):
    with pytest.raises(ValueError, match='PROFILE=nsys captures the timed sweep'):
        cards.generate(entry, 'a', dict(instance='p', slot='s'), tmp_path, tmp_path / 'key', {'PROFILE': 'nsys'}, set(), None, 1)
    job = cards.generate({**entry, 'levels': [1]}, 'a', dict(instance='p', slot='s'), tmp_path, tmp_path / 'key',
                         {'PROFILE': 'nsys,draft-timing'}, set(), None, 1)
    window = job['entry']['window']
    coordinator = 'cuteafd-coordinator-' + job['entry']['set']['INSTANCE']
    assert window['open']['argv'] == ['docker', 'exec', coordinator, '/usr/local/cuda/bin/nsys', 'start', '--session=cuteafd-profile']
    assert window['close']['argv'][-2:] == ['stop', '--session=cuteafd-profile']
    assert window['open']['required'] and window['close']['required']
    matrix = json.loads((Path(job['state']) / 'matrix.json').read_text())
    assert matrix['entries'][0]['window'] == window
    # Other switches add no window.
    plain = cards.generate({**entry, 'levels': [1]}, 'b', dict(instance='q', slot='s'), tmp_path, tmp_path / 'key',
                           {'PROFILE': 'expert-timing'}, set(), None, 1)
    assert 'window' not in plain['entry']


def test_session_name_matches_the_launcher():
    text = (ROOT / 'scripts/lib/release-common.sh').read_text()
    assert f'RELEASE_NSYS_SESSION={cards.NSYS_SESSION}\n' in text


def test_profiler_reports_are_summarized_by_name_and_size(entry, tmp_path, monkeypatch):
    entry = {**entry, 'sparks': [], 'set': {**entry['set'], 'SPARK_COUNT': '0'}, 'levels': [1], 'profile': 'panels:'}
    job = cards.generate(entry, 'a', dict(instance='p', slot='s'), tmp_path, tmp_path / 'key', {'PROFILE': 'nsys'}, set(), None, 1)
    profile = Path(job['entry']['set']['PROFILE_DIR'])
    sweep = Path(job['state']) / 'reports' / 'sweep-only' / job['entry']['name'] / 'sweep.json'
    sweep.parent.mkdir(parents=True)
    sweep.write_text(json.dumps({'points': [{'c': 1, 'phase': 'levels', 'aggregate_tok_s': 50.0}]}))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'failed' and 'produced no .nsys-rep report' in row['error']
    (profile / 'nsys-raptor.nsys-rep').write_bytes(b'x' * 3 * 2**20)
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'pass', row
    assert row['profile']['reports'] == [{'name': 'nsys-raptor.nsys-rep', 'bytes': 3 * 2**20}]
    cards.write_summary(tmp_path, [row], {'a': {}})
    assert 'nsys-raptor.nsys-rep (3.0 MiB)' in (tmp_path / 'summary.md').read_text()


def test_sweep_rows_and_sweep_only_jobs(entry, tmp_path):
    entry = {**entry, 'profile': 'panels:', 'levels': [1, 16], 'post_load_c1': True}
    job = make(entry, tmp_path)
    sweep = Path(job['state']) / 'reports' / 'sweep-only' / job['entry']['name'] / 'sweep.json'
    sweep.parent.mkdir(parents=True)
    sweep.write_text(json.dumps({'points': [{'c': 1, 'phase': 'levels', 'aggregate_tok_s': 50.0},
        {'c': 16, 'phase': 'levels', 'aggregate_tok_s': 400.0}, {'c': 1, 'phase': 'post_load', 'aggregate_tok_s': 47.5}]}))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'pass', row
    assert (row['sweep C1'], row['sweep C16'], row['post-load C1']) == (50.0, 400.0, 47.5)
    assert row.get('C1') is None  # the baseline's C1 is a different measurement


def test_binary_without_sweep_fields_is_refused(entry, tmp_path, monkeypatch):
    job = make(entry, tmp_path)
    monkeypatch.setattr(cards.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stderr='  v41 family=x set={}', stdout=''))
    with pytest.raises(ValueError, match='predates smoke fields levels, post_load_c1'):
        cards.binary_supports('/old/cuteafd', ROOT, job, ['levels', 'post_load_c1'])
    monkeypatch.setattr(cards.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stderr='levels=Some([1]) post_load_c1=true', stdout=''))
    cards.binary_supports('/new/cuteafd', ROOT, job, ['levels', 'post_load_c1'])


def test_build_config_is_staging_only_with_card_overrides_and_tagged_images(tmp_path, monkeypatch):
    # #72: a 4-Spark MiMo card was staged on six hosts with SPARK_TP=6 forced into
    # the build config, its --set overrides were ignored and untagged dev images
    # were used. Export scope comes from the card; TP never from the staging set.
    kit = Path.home() / '.cache/cuteafd/builds/release-v2-rc3/kit'
    if not (kit / 'matrix-main.json').exists():
        pytest.skip('release-v2-rc3 kit absent')
    card = cards.load_cards(kit, ['mimo26-flash-max'])[0]
    card['set']['RTX_EXPERT_LAYERS'] = '0'
    arms = cards.arms_from(['baseline=x:a', 'candidate=y:b'])
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    cards.build_arms(['baseline=HEAD', 'candidate=HEAD'], arms, [card], tmp_path, 'task', True)
    cfg = cards.config(tmp_path / 'build-candidate.config')
    assert 'SPARK_TP' not in cfg and 'SPARK_EP' not in cfg and 'MODEL_VARIANT' not in cfg
    assert cfg['RTX_EXPERT_LAYERS'] == '0' and cfg['MODEL_ID'].startswith('XiaomiMiMo/')
    tag = 'tc-' + subprocess.run([sys.executable, str(ROOT / 'scripts/build/dev-toolchain.py')], capture_output=True, text=True).stdout.strip()
    assert cfg['COORDINATOR_DOCKER_DEV'] == 'cuteafd-coordinator-dev:' + tag
    assert cfg['SPARK_EXPERT_DOCKER_DEV'] == 'cuteafd-spark-expert-dev:' + tag
    scopes = cards.build_scopes([card])
    assert scopes['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == 'mimof=4'
    assert scopes['CUTEAFD_WIP_AUDIO_AOT'] == 'ON'
    assert cards.slot_needs(card) == {'coordinator': ['fp8/fp8-mimof'], 'spark': ['fp8/fp8-mimof'], 'audio': True,
                                      'tp_role': None, 'nvfp4': False}


def _fake_slot(packages=(), audio=True):
    return lambda arm, host: json.dumps(dict(slot=arm['slot'], seal_sha256='seal', artifact_manifest_sha256='art',
                                             contents=dict(packages=list(packages), audio=audio)))


def _run_main(monkeypatch, argv, slot=None, rc_for=lambda job: 0, results=None):
    """Runs main() with hardware stubbed: no containers, a sealed slot, and a
    smoke 'process' that writes an exit code and (optionally) a result."""
    monkeypatch.setattr(sys, 'argv', ['wip-cards', *argv])
    monkeypatch.setattr(cards, 'assert_absent', lambda job: None)
    monkeypatch.setattr(cards, 'slot_check', slot or _fake_slot())
    monkeypatch.setattr(cards, 'binary_supports', lambda *a, **k: None)
    launched = []
    def detached(cmd, log, exit_file, env):
        job = json.loads((exit_file.parent / 'job.json').read_text())
        launched.append(job['card'] + '/' + job['arm'])
        exit_file.write_text(f'{rc_for(job)}\n')
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(cards, 'detached', detached)
    def summarize(job, code):
        row = dict(card=job['card'], arm=job['arm'], repeat=job['repeat'], simulated=False,
                   status='pass' if code == 0 else 'failed', exit_code=code)
        if code:
            row.update(error='launcher said no', failure_kind='launch')
        cards.save(Path(job['state']) / 'result.json', row)
        return row
    monkeypatch.setattr(cards, 'summarize_job', summarize)
    return cards.main(), launched


def test_multi_arm_without_mode_is_refused_with_a_clear_message(entry, tmp_path, monkeypatch, capsys):
    kit = _kit(tmp_path, entry)
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--arm', 'a=b:s', '--arm', 'c=d:s', '--dry-run'])
    with pytest.raises(SystemExit):
        cards.main()
    assert '--interleave' in capsys.readouterr().err


def test_resume_keeps_passed_jobs_and_reruns_failed_keeping_old_diagnosis(entry, tmp_path, monkeypatch):
    kit = _kit(tmp_path, entry, {**entry, 'name': 'other'})
    state = tmp_path / 'state'
    argv = ['--kit', str(kit), '--cards', 'v41-min', 'other', '--matrix', '--arm', 'a=b:s', '--state', str(state)]
    rc, launched = _run_main(monkeypatch, argv, rc_for=lambda job: 1 if job['card'] == 'other' else 0)
    assert rc == 1 and sorted(launched) == ['other/a', 'v41-min/a']
    # A plain retry is refused, never silently mixed with the earlier run.
    with pytest.raises(ValueError, match='--resume'):
        _run_main(monkeypatch, argv)
    rc, launched = _run_main(monkeypatch, argv + ['--resume'])
    assert rc == 0 and launched == ['other/a']
    other = next(state.glob('other-a-r1*'))
    assert json.loads((other / 'attempt-1' / 'result.json').read_text())['error'] == 'launcher said no'
    assert json.loads((other / 'result.json').read_text())['status'] == 'pass'
    summary = json.loads((state / 'summary.json').read_text())
    assert sorted(r['status'] for r in summary['rows']) == ['pass', 'pass']


def test_default_state_is_fresh_when_the_default_task_collides(entry, tmp_path, monkeypatch):
    kit = _kit(tmp_path, entry)
    monkeypatch.setenv('HOME', str(tmp_path))
    builds = tmp_path / '.cache/cuteafd/builds'
    argv = ['--kit', str(kit), '--cards', 'v41-min', '--matrix', '--arm', 'a=b:s']
    _run_main(monkeypatch, argv)
    first = [p for p in builds.iterdir() if p.name.startswith('cards-')]
    assert len(first) == 1
    # Same default name again (a same-second rerun): a fresh state, not a refusal.
    original = cards.argparse.ArgumentParser.parse_args
    def same_task(self, *a, **k):
        args = original(self, *a, **k)
        args.task = first[0].name
        return args
    monkeypatch.setattr(cards.argparse.ArgumentParser, 'parse_args', same_task)
    _run_main(monkeypatch, argv)
    assert (builds / (first[0].name + '-2')).is_dir()
    assert list((builds / (first[0].name + '-2')).glob('*/exit'))


def test_preflight_refuses_missing_packages_and_audio_before_any_launch(tmp_path, monkeypatch):
    kit = Path.home() / '.cache/cuteafd/builds/release-v2-rc3/kit'
    gate = ROOT / 'scripts/bench/kits/gate'
    argv = ['--kit', str(gate), '--cards', 'mimo26-flash-max', 'glmf-exl3-max', '--matrix', '--arm', 'c=x:s',
            '--state', str(tmp_path / 'state')]
    with pytest.raises(ValueError) as caught:
        _run_main(monkeypatch, argv, slot=_fake_slot(packages=['fp8/fp8-mimof'], audio=False))
    message = str(caught.value)
    assert 'preflight failed before queuing' in message
    assert 'built without the audio tower (AUDIO=auto)' in message
    assert 'lacks exl3/exl3-glmf-k34, fp8/fp8-glmf' in message
    assert not list((tmp_path / 'state').glob('*/exit'))
    assert json.loads((tmp_path / 'state' / 'preflight.json').read_text())['status'] == 'failed'
    rc, launched = _run_main(monkeypatch, argv + ['--resume'], slot=_fake_slot(
        packages=['fp8/fp8-mimof', 'exl3/exl3-glmf-k34', 'fp8/fp8-glmf'], audio=True))
    assert rc == 0 and sorted(launched) == ['glmf-exl3-max/c', 'mimo26-flash-max/c']


def test_preflight_only_and_grant_hook(tmp_path, monkeypatch, capsys):
    gate = ROOT / 'scripts/bench/kits/gate'
    state = tmp_path / 'state'
    argv = ['--kit', str(gate), '--cards', 'qwen38-exl3-min', '--matrix', '--arm', 'c=x:s', '--state', str(state), '--preflight-only']
    rc, launched = _run_main(monkeypatch, argv, slot=_fake_slot(packages=['exl3/exl3-qwen4-k45', 'fp8/fp8-qwen4-nvfp4']))
    assert rc == 0 and launched == []
    job = json.loads((state / 'qwen38-exl3-min-c-r1' / 'job.json').read_text())
    hook = job['entry']['prelaunch']['argv']
    assert hook[-2] == '--preflight-job' and job['entry']['prelaunch']['required'] is True
    # At grant: same checks; a rebuilt slot (different seal) is refused.
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--preflight-job', hook[-1]])
    assert cards.main() == 0
    monkeypatch.setattr(cards, 'slot_check', lambda arm, host: json.dumps(dict(slot='s', seal_sha256='other', artifact_manifest_sha256='art',
                                                                         contents=dict(packages=['exl3/exl3-qwen4-k45', 'fp8/fp8-qwen4-nvfp4'], audio=True))))
    with pytest.raises(ValueError, match='sealed artifacts changed since enqueue'):
        cards.main()


def test_trace_is_per_job_owned_and_checked(entry, tmp_path):
    job = cards.generate(entry, 'a', dict(instance='t', slot='s'), tmp_path, tmp_path / 'key', {}, set(), None, 1, trace=True)
    trace = Path(job['state']) / 'speculation-trace.jsonl'
    assert job['entry']['set']['SPECULATION_TRACE'] == str(trace)
    assert job['entry']['observer']['argv'][-2] == '--observe'
    with pytest.raises(ValueError, match='--trace owns SPECULATION_TRACE'):
        cards.generate(entry, 'a', dict(instance='t', slot='s'), tmp_path, tmp_path / 'key', {'SPECULATION_TRACE': '/x'}, set(), None, 1, trace=True)
    report = Path(job['state']) / 'reports' / 'one' / 'report.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'baseline': {'quality': {'checks': [], 'status': 'pass'}}}))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'failed' and row['failure_kind'] == 'measurement'
    assert 'speculation trace is empty' in row['error']
    trace.write_text('{"type":"draft_round"}\n{"type":"draft_round"}\n')
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'pass' and row['trace']['records'] == 2 and row['trace']['path'] == str(trace)
    off = cards.generate(entry, 'b', dict(instance='u', slot='s'), tmp_path, tmp_path / 'key', {'SPECULATOR': 'off'}, set(), None, 1, trace=True)
    (Path(off['state']) / 'reports' / 'one').mkdir(parents=True)
    (Path(off['state']) / 'reports' / 'one' / 'report.json').write_text(report.read_text())
    row = cards.summarize_job(off, 0)
    assert row['status'] == 'pass' and row['trace']['expected'] is False


def test_arm_specific_config_on_one_build_end_to_end(tmp_path, monkeypatch):
    # Two arms on one sealed build that differ only by placement (gap 5).
    gate = ROOT / 'scripts/bench/kits/gate'
    state = tmp_path / 'state'
    argv = ['--kit', str(gate), '--cards', 'mimo26-flash-max', '--interleave', '--repeats', '2',
            '--arm', 'tp2=m-a:s', '--arm', 'zero=m-b:s', '--arm-wip', 'tp2=shared', '--arm-wip', 'zero=shared',
            '--arm-set', 'tp2:mimo26-flash-max:RTX_EXPERT_LAYERS=41', '--arm-set', 'zero:mimo26-flash-max:RTX_EXPERT_LAYERS=0',
            '--state', str(state), '--dry-run']
    _run_main(monkeypatch, argv)
    jobs = json.loads((state / 'plan.json').read_text())['jobs']
    assert [j['arm'] for j in jobs] == ['tp2', 'zero', 'zero', 'tp2']
    for job in jobs:
        values = cards.config(Path(job['state']) / 'base.config')
        assert values['RTX_EXPERT_LAYERS'] == ('41' if job['arm'] == 'tp2' else '0')
        assert values['WIP_INSTANCE'] == 'shared' and job['entry']['run_args'] == ['--wip', 's']
    assert len({j['entry']['set']['INSTANCE'] for j in jobs}) == 4


def test_first_real_error_wins_and_kinds_are_distinguished(entry, tmp_path):
    job = make(entry, tmp_path)
    dest = Path(job['state'])
    logs = dest / 'smoke' / 'logs'
    logs.mkdir(parents=True)
    name = job['entry']['name']
    (logs / (name + '.log')).write_text('starting\ndodo expert worker exited:\n'
        'Error: audio library: /opt/cuteafd/lib/libcuteafd_native.so: undefined symbol: cuteafd_audio_required\n'
        'error: ./run.sh exited exit status: 1 (log x)\nError response from daemon: No such container: c\n')
    (dest / 'smoke' / 'state.json').write_text(json.dumps({name: {'status': 'failed', 'error': './run.sh exited exit status: 1 (log x)'}}))
    job['nonce_seed'] = 'seed'
    row = cards.summarize_job(job, 1)
    assert row['failure_kind'] == 'launch'
    assert row['error'] == 'dodo expert worker exited:' or 'undefined symbol: cuteafd_audio_required' in row['error']
    assert row['errors'][0] == row['error'] and any('./run.sh exited' in e for e in row['errors'][1:])
    # A later derived check (the prompt hash) never replaces it.
    cards.validate_matched_pairs([{**row, 'nonce_seed': 'seed', 'state': str(dest)}], ['baseline'])
    assert json.loads((dest / 'result.json').read_text())['error'] == row['error']
    # Served, then a correctness check failed: quality, with the check's own words.
    job2 = make(entry, tmp_path, arm='q', instance='q')
    report = Path(job2['state']) / 'reports' / 'one' / 'report.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'baseline': {'quality': {'status': 'fail', 'checks': [
        {'id': 'spec_lossless', 'status': 'fail', 'summary': 'greedy drafts-on/off token IDs differ at 12'}]}}}))
    row = cards.summarize_job(job2, 0)
    assert row['failure_kind'] == 'quality' and 'spec_lossless: greedy drafts-on/off token IDs differ' in row['error']


def test_summary_md_has_before_after_ratios_and_failures(tmp_path):
    rows = [dict(card='c', arm=arm, repeat=r, status='pass', simulated=False, C1=c1, **{'sweep C16': c16, 'post-load C1': p})
            for r, (arm, c1, c16, p) in enumerate([('base', 100.0, 800.0, 95.0), ('cand', 110.0, 880.0, 100.0)], 1)]
    rows[1]['repeat'] = 1
    rows.append(dict(card='d', arm='cand', repeat=1, status='failed', simulated=False, error='undefined symbol: x', failure_kind='launch'))
    cards.write_summary(tmp_path, rows, {'base': {}, 'cand': {}})
    text = (tmp_path / 'summary.md').read_text()
    assert '## Before -> after' in text
    assert '| c | C1 | 100 | 110 | 1.1 | 10 | 1 |' in text
    assert '| c | sweep C16 | 800 | 880 | 1.1 |' in text and 'post-load C1' in text
    assert '**d / cand / r1** (launch): undefined symbol: x' in text


def test_panels_alone_skip_the_baseline_and_prefill_lengths_reach_the_sweep(entry, tmp_path, monkeypatch):
    kit = _kit(tmp_path, entry)
    state = tmp_path / 'state'
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix', '--arm', 'a=b:s',
        '--state', str(state), '--panels', 'prefill', '--prefill-lengths', '8192', '--dry-run'])
    assert cards.main() == 0
    e = json.loads(next(state.glob('*/matrix.json')).read_text())['entries'][0]
    assert e['profile'] == 'panels:prefill' and e['baseline'] is False and e['prefill_lengths'] == [8192]
    for listed, expected in [('baseline,prefill', None), ('none', None)]:
        state2 = tmp_path / ('s-' + listed.replace(',', '-'))
        monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix', '--arm', 'a=b:s',
            '--state', str(state2), '--panels', listed, '--prefill-lengths', '8192', '--dry-run'])
        assert cards.main() == 0
        assert json.loads(next(state2.glob('*/matrix.json')).read_text())['entries'][0].get('baseline') is expected


def test_prefill_sweep_rows(entry, tmp_path):
    entry = {**entry, 'profile': 'panels:', 'prefill_lengths': [8192]}
    job = make(entry, tmp_path)
    sweep = Path(job['state']) / 'reports' / 'sweep-only' / job['entry']['name'] / 'sweep.json'
    sweep.parent.mkdir(parents=True)
    sweep.write_text(json.dumps({'points': [], 'prefill': [{'length': 8192, 'prompt_tokens': 8170, 'tok_s': 9000.0, 'ttft_s': 0.91}]}))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'pass' and row['prefill 8192'] == 9000.0 and row['prefill 8192 TTFT s'] == 0.91


def test_build_scopes_are_per_card_zero_spark_and_family_counts():
    base = {'config': '/dev/null'}
    local = {**base, 'name': 'qwen38-exl3-min', 'family': 'qwen4', 'sparks': [], 'set': {'SPARK_COUNT': '0'}}
    glm = {**base, 'name': 'glm53-exl3-max', 'family': 'glm5', 'sparks': list(cards.HOSTS), 'set': {'SPARK_COUNT': '6'}}
    mixed = cards.build_scopes([glm, local])
    spark = set(mixed['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';'))
    assert not any(scope.startswith('qwen4:') for scope in spark)
    assert 'qwen4:exl3-k45' in mixed['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert mixed['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == 'glm=6'
    assert mixed['CUTEAFD_WIP_QWEN4_AOT'] == 'ON'
    qwen4 = {**local, 'sparks': list(cards.HOSTS[:4]), 'set': {'SPARK_COUNT': '4'}}
    both = cards.build_scopes([glm, qwen4])
    assert 'qwen4:exl3-k45' in both['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')
    assert both['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == 'glm=6;qwen4=4'


def _slot_with_roles(roles, verified=True, nvfp4=(), error=None):
    return lambda arm, host: json.dumps(dict(slot=arm['slot'], seal_sha256='seal', artifact_manifest_sha256='art',
        contents=dict(packages=[], audio=True, tp_roles=None if roles is None else dict(
            roles=list(roles), verified=verified, nvfp4=list(nvfp4), error=error))))


@pytest.mark.parametrize('roles,verified,error,message', [
    # The #98 slot: built for the TP4 set, no extra roles.
    ([], False, None, 'attests Spark TP roles <none>, not tp3'),
    (['tp2', 'tp6'], True, None, 'attests Spark TP roles tp2;tp6, not tp3'),
    (['tp3'], False, None, 'TP symbols are not verified for tp3'),
    (None, True, None, 'has no V41_EXPERT_TP_AOT.json'),
    ([], False, 'TP manifest hash mismatch', 'cannot attest Spark TP role tp3: TP manifest hash mismatch'),
])
def test_preflight_validates_attested_spark_tp_roles_before_enqueue(tmp_path, monkeypatch, roles, verified, error, message):
    """#98: the card-derived Spark TP (v41-flash-min: SPARK_TP=3) must be attested by
    every rank's slot before enqueue, as the launcher refuses it at launch."""
    gate = ROOT / 'scripts/bench/kits/gate'
    argv = ['--kit', str(gate), '--cards', 'v41-flash-min', '--matrix', '--arm', 'c=x:s',
            '--state', str(tmp_path / 'state')]
    with pytest.raises(ValueError) as caught:
        _run_main(monkeypatch, argv, slot=_slot_with_roles(roles, verified, error=error))
    text = str(caught.value)
    assert 'preflight failed before queuing' in text and message in text
    for host in ('ostrich', 'dodo', 'emu'):
        assert f'{host}: slot s' in text
    assert 'CUTEAFD_WIP_SPARK_TP_ROLES=tp3' in text
    assert not list((tmp_path / 'state').glob('*/exit'))
    # A slot that attests tp3 with verified symbols launches.
    rc, launched = _run_main(monkeypatch, argv + ['--resume'], slot=_slot_with_roles(['tp3']))
    assert rc == 0 and launched == ['v41-flash-min/c']


def test_attested_spark_tp_roles_are_checked_again_at_grant(tmp_path, monkeypatch):
    gate = ROOT / 'scripts/bench/kits/gate'
    state = tmp_path / 'state'
    argv = ['--kit', str(gate), '--cards', 'v41-flash-min', '--matrix', '--arm', 'c=x:s', '--state', str(state),
            '--preflight-only']
    rc, launched = _run_main(monkeypatch, argv, slot=_slot_with_roles(['tp2', 'tp3', 'tp6']))
    assert rc == 0 and launched == []
    job = json.loads((state / 'v41-flash-min-c-r1' / 'job.json').read_text())
    assert job['needs']['tp_role'] == 'tp3'
    hook = job['entry']['prelaunch']['argv']
    # At grant the slot no longer attests tp3 (rebuilt or swapped): refused.
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--preflight-job', hook[-1]])
    monkeypatch.setattr(cards, 'slot_check', _slot_with_roles(['tp2']))
    with pytest.raises(ValueError, match='attests Spark TP roles tp2, not tp3'):
        cards.main()


@pytest.mark.parametrize('keys,role', [
    ({'SPARK_COUNT': '4'}, None),
    ({'SPARK_COUNT': '3', 'SPARK_TP': '3', 'SPARK_EP': '1'}, 'tp3'),
    ({'SPARK_COUNT': '6', 'SPARK_TP': '6', 'SPARK_EP': '1'}, 'tp6'),
    ({'SPARK_COUNT': '4', 'SPARK_TP': '2', 'SPARK_EP': '2'}, 'tp2'),
    ({'SPARK_COUNT': '4', 'SPARK_TP': '4', 'SPARK_EP': '1'}, None),
    # A native count other than 4 is TP=count at launch.
    ({'SPARK_COUNT': '6'}, 'tp6'),
    ({'SPARK_COUNT': '2', 'EXPERT_FORMAT': 'exl3'}, None),
    ({'SPARK_COUNT': '0'}, None),
])
def test_v41_tp_role_matches_the_launcher_derivation(tmp_path, keys, role):
    config = tmp_path / 'c.config'
    config.write_text('MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash\n')
    card = dict(name='v', family='deepseek_v41', config=str(config), set=keys, sparks=[])
    assert cards.slot_needs(card)['tp_role'] == role
    assert cards.slot_needs({**card, 'family': 'glm5_flash'})['tp_role'] is None


def test_seal_check_reports_attested_roles_from_the_slot(tmp_path, monkeypatch):
    import hashlib
    root = tmp_path / 'wip-inst'
    slot = _sealed_slot(root, role='spark-expert')
    art = slot / 'workspace/.cuteafd-wip'
    (art / 'libcuteafd_native.so').write_bytes(b'lib')
    manifest = json.dumps(dict(schema=1, spark_tp_roles=['tp3'], symbols_verified=True,
                               native_library_sha256=hashlib.sha256(b'lib').hexdigest(),
                               nvfp4_spark_tp_roles=[])).encode()
    (art / 'V41_EXPERT_TP_AOT.json').write_bytes(manifest)
    meta = json.loads((slot / 'META.json').read_text())
    meta.update(spark_tp_roles=['tp3'], v41_expert_tp_manifest_sha256=hashlib.sha256(manifest).hexdigest())
    data = json.dumps(meta).encode()
    (slot / 'META.json').write_bytes(data)
    (slot / 'FINGERPRINT').write_text(hashlib.sha256(data).hexdigest() + '\n')
    # Spark hosts read the spark-expert role; run the seal script as each host does.
    script = cards.SLOT_SEAL_CHECK
    result = subprocess.run(['python3', '-c', script, 's', 'spark-expert', 'inst', 'none', str(root)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    roles = json.loads(result.stdout)['contents']['tp_roles']
    assert roles == dict(roles=['tp3'], verified=True, nvfp4=[], error=None)
    # Every seal check reports the field, so preflight can never skip it.
    (art / 'V41_EXPERT_TP_AOT.json').unlink()
    result = subprocess.run(['python3', '-c', script, 's', 'spark-expert', 'inst', 'none', str(root)],
                            capture_output=True, text=True)
    assert json.loads(result.stdout)['contents']['tp_roles'] is None
    (art / 'V41_EXPERT_TP_AOT.json').write_bytes(manifest)
    # A library that no longer matches its manifest cannot attest a role.
    (art / 'libcuteafd_native.so').write_bytes(b'other')
    result = subprocess.run(['python3', '-c', script, 's', 'spark-expert', 'inst', 'none', str(root)],
                            capture_output=True, text=True)
    assert json.loads(result.stdout)['contents']['tp_roles']['error'] == 'native library hash mismatch'


def test_lossless_set_panel_and_file_reach_the_smoke_entry(entry, tmp_path, monkeypatch):
    kit = _kit(tmp_path, entry)
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps([{'id': 'glmf-v19-base', 'prompt': 'Write code.', 'nonce': '7ec2b2071ff95934-0000000026',
                                  'source': 'drafter-efficiency-serving-v19', 'reconstructed': True}]))
    state = tmp_path / 'state'
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix', '--arm', 'a=b:s',
        '--state', str(state), '--panels', 'baseline,spec_lossless_set', '--lossless-set', str(cases), '--dry-run'])
    assert cards.main() == 0
    e = json.loads(next(state.glob('*/matrix.json')).read_text())['entries'][0]
    assert e['profile'] == 'panels:baseline,spec_lossless_set' and e['lossless_set'] == str(cases.resolve())
    assert 'baseline' not in e  # Listed baseline keeps the card's baseline (and its single spec_lossless).
    # Without the panel the file has nothing to feed.
    monkeypatch.setattr(sys, 'argv', ['wip-cards', '--kit', str(kit), '--cards', 'v41-min', '--matrix', '--arm', 'a=b:s',
        '--state', str(tmp_path / 's2'), '--panels', 'baseline', '--lossless-set', str(cases), '--dry-run'])
    with pytest.raises(SystemExit):
        cards.main()


@pytest.mark.parametrize('cases, message', [
    ([], '1 to 64 cases'),
    ({'id': 'a'}, '1 to 64 cases'),
    ([{'id': 'a', 'prompt': 'p'}], 'string id, prompt and nonce'),
    ([{'id': 'a', 'prompt': 'p', 'nonce': 'n', 'extra': 1}], 'string id, prompt and nonce'),
    ([{'id': 'a b', 'prompt': 'p', 'nonce': 'n'}], 'invalid or duplicate case id'),
    ([{'id': 'a', 'prompt': 'p', 'nonce': 'n'}, {'id': 'a', 'prompt': 'q', 'nonce': 'm'}], 'invalid or duplicate case id'),
    ([{'id': 'a', 'prompt': 'p', 'nonce': 'x]'}], 'brackets or control'),
])
def test_malformed_lossless_sets_are_refused_before_hardware(tmp_path, cases, message):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match=message):
        cards.load_lossless_set(path)


def test_shipped_lossless_set_is_valid():
    path = cards.REPO / 'scripts/bench/fixtures/spec-lossless-set-v1.json'
    assert cards.load_lossless_set(path) == str(path)
    cases = json.loads(path.read_text())
    assert len({c['nonce'] for c in cases}) == len(cases)
    assert next(c for c in cases if c['id'] == 'glmf-v19-base')['reconstructed'] is True


def test_lossless_set_gate_fails_the_row_and_names_failing_cases(entry, tmp_path):
    job = make({**entry, 'profile': 'panels:baseline,spec_lossless_set'}, tmp_path)
    report_dir = Path(job['state']) / 'reports' / 'one'
    report_dir.mkdir(parents=True)
    rows = [{'id': 'code-1', 'pass': True}, {'id': 'code-2', 'pass': False}]
    panel = {'id': 'spec_lossless_set', 'status': 'done', 'passes': [{'gate': 'fail', 'passed': 1, 'total': 2,
             'summary': 'dflash: 1/2 lossless; code-2: ids identical, but 1 of 128 rows differ from token 40', 'rows': rows,
             'set': {'name': 'spec-lossless-set-v1'}, 'evidence': {'code-2': {'nonce': '002888898b'}}}]}
    report = {'baseline': {'quality': {'status': 'pass', 'checks': [{'id': 'spec_lossless', 'status': 'pass'}]}}, 'panels': [panel]}
    (report_dir / 'report.json').write_text(json.dumps(report))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'failed' and row['failure_kind'] == 'quality'
    assert row['lossless_set']['failing'] == ['code-2'] and 'code-2: ids identical' in row['error']
    assert 'evidence' not in row['lossless_set']
    assert cards.spec_cell(row) == 'pass; set 1/2 fail (code-2)'
    panel['passes'][0].update(gate='pass', passed=2)
    rows[1]['pass'] = True
    (report_dir / 'report.json').write_text(json.dumps(report))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'pass' and row['lossless_set']['status'] == 'pass'
    # A panel that errored before any pass is a failure, not a silent skip.
    panel.update(status='failed', passes=[], error='drafts on: HTTP 500')
    (report_dir / 'report.json').write_text(json.dumps(report))
    row = cards.summarize_job(job, 0)
    assert row['status'] == 'failed' and 'HTTP 500' in row['error']
    # No speculator: the server marks the panel unsupported.
    panel.update(status='unsupported', error='no speculator: nothing to check')
    (report_dir / 'report.json').write_text(json.dumps(report))
    assert cards.summarize_job(job, 0)['lossless_set']['status'] == 'skipped'
