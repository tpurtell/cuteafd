"""scripts/dev/cpu-gate: verdicts from parsed test summaries, never exit codes alone."""
import contextlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOADER = importlib.machinery.SourceFileLoader('cpu_gate', str(ROOT / 'scripts/dev/cpu-gate'))
SPEC = importlib.util.spec_from_loader('cpu_gate', LOADER)
gate = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(gate)

BUILD_OK = '{"reason":"build-finished","success":true}\n'
REAL_PLACEMENT_INPUTS = gate.placement_inputs


@pytest.fixture(autouse=True)
def matched_placement_inputs(monkeypatch):
    """Suites under test here are not about placement inputs; test_placement_* use the real resolver."""
    monkeypatch.setattr(gate, 'placement_inputs', lambda source, env, args: ({'stub': True}, None))

ARGS = SimpleNamespace(cargo_build_timeout=60, cargo_timeout=60, scripts_timeout=60, skip_live_planners=False, cargo_jobs=4, pytest_jobs=4)


def _fake(monkeypatch, outputs):
    """outputs: list of (exit, text) in call order for run_logged."""
    calls = []
    def run_logged(cmd, log_path, env, cwd, timeout):
        code, text = outputs[len(calls)]
        calls.append(cmd)
        Path(log_path).write_text(text)
        return code, 1.0
    monkeypatch.setattr(gate, 'compiler_cache_env', lambda source, root, env: dict(env))
    monkeypatch.setattr(gate, 'nextest_for', lambda env: None)
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/gate-venv/bin/python'))
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    monkeypatch.setattr(gate, 'validate_full_test_ids', lambda *a: {})
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: '')
    return calls


def test_missing_toolchain_refuses_before_any_work(tmp_path, monkeypatch, capsys):
    """With no candidate location supplying cargo/rustc, the gate still refuses
    before touching the build directory."""
    monkeypatch.setattr(gate, 'BUILDS', tmp_path / 'builds')
    monkeypatch.setattr(gate, 'HOMEBREW_BIN', tmp_path / 'no-homebrew')
    monkeypatch.setattr(gate, 'RUSTUP_HOME', tmp_path / 'no-rustup')
    monkeypatch.setattr(sys, 'argv', ['cpu-gate', '--task', 'path-probe'])
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('PATH', '/nonexistent')
    monkeypatch.delenv('CARGO_HOME', raising=False)
    with pytest.raises(SystemExit) as refused:
        gate.main()
    assert refused.value.code == 2
    assert 'cargo, rustc not on PATH' in capsys.readouterr().err
    assert not (tmp_path / 'builds/path-probe').exists()


def test_toolchain_path_recovers_cargo_from_the_standard_locations(tmp_path, monkeypatch):
    """A systemd unit starts without ~/.cargo/bin; the gate must find a stub there
    instead of refusing, and leave a working PATH alone."""
    stub = tmp_path / '.cargo/bin'
    stub.mkdir(parents=True)
    for tool in gate.TOOLCHAIN:
        (stub / tool).write_text('#!/bin/sh\n')
        (stub / tool).chmod(0o755)
    path, used = gate.toolchain_path({'HOME': str(tmp_path), 'PATH': ''})
    assert used == str(stub)
    assert path.split(os.pathsep)[0] == str(stub)
    assert shutil.which('cargo', path=path) == str(stub / 'cargo')
    assert shutil.which('rustc', path=path) == str(stub / 'rustc')
    # A working PATH is returned untouched, whatever the standard locations hold.
    path, used = gate.toolchain_path({'HOME': str(tmp_path), 'PATH': str(stub)})
    assert (path, used) == (str(stub), None)
    # $CARGO_HOME wins over ~/.cargo, which still comes before the persistent home.
    cargo_home = tmp_path / 'cargo-home'
    (cargo_home / 'bin').mkdir(parents=True)
    for tool in gate.TOOLCHAIN:
        (cargo_home / 'bin' / tool).write_text('#!/bin/sh\n')
        (cargo_home / 'bin' / tool).chmod(0o755)
    path, used = gate.toolchain_path({'HOME': str(tmp_path), 'CARGO_HOME': str(cargo_home), 'PATH': ''})
    assert used == str(cargo_home / 'bin') and path.split(os.pathsep)[0] == str(cargo_home / 'bin')
    assert path.split(os.pathsep)[1] == str(stub)
    # Homebrew's prefix and the toolchain rustup resolves follow those three.
    brew, rustup_bin = tmp_path / 'homebrew/bin', tmp_path / 'rustup/toolchains/1.99.1/bin'
    for directory in (brew, rustup_bin):
        directory.mkdir(parents=True)
        for tool in gate.TOOLCHAIN:
            (directory / tool).write_text('#!/bin/sh\n')
            (directory / tool).chmod(0o755)
    empty = {'HOME': str(tmp_path / 'empty'), 'PATH': '/nonexistent'}
    monkeypatch.setattr(gate, 'HOMEBREW_BIN', brew)
    path, used = gate.toolchain_path(empty)
    assert used == str(brew) and path.split(os.pathsep)[0] == str(brew)
    monkeypatch.setattr(gate, 'HOMEBREW_BIN', tmp_path / 'no-homebrew')
    monkeypatch.setattr(gate, 'rustup_toolchain_bin', lambda root=gate.RUSTUP_HOME, env=None: rustup_bin)
    path, used = gate.toolchain_path(empty)
    assert used == str(rustup_bin) and path.split(os.pathsep)[0] == str(rustup_bin)
    # Nothing to add: unchanged and reported as not found, so main still refuses.
    monkeypatch.setattr(gate, 'rustup_toolchain_bin', lambda root=gate.RUSTUP_HOME, env=None: None)
    assert gate.toolchain_path({'HOME': str(tmp_path / 'empty'), 'PATH': '/nonexistent'}) == ('/nonexistent', None)


def test_rustup_toolchain_bin_asks_rustup_for_its_toolchain(tmp_path):
    toolchain = tmp_path / 'toolchains/1.99.1/bin'
    toolchain.mkdir(parents=True)
    cargo = toolchain / 'cargo'
    cargo.write_text('#!/bin/sh\n')
    cargo.chmod(0o755)
    root = tmp_path / 'rustup'
    (root / 'bin').mkdir(parents=True)
    rustup = root / 'bin/rustup'
    rustup.write_text(f'#!/bin/sh\n[ "$1" = which ] && echo {cargo}\n')
    rustup.chmod(0o755)
    assert gate.rustup_toolchain_bin(root=root, env={'PATH': ''}) == toolchain
    assert gate.rustup_toolchain_bin(root=tmp_path / 'absent') is None
    rustup.write_text('#!/bin/sh\nexit 1\n')
    assert gate.rustup_toolchain_bin(root=root, env={'PATH': ''}) is None
    rustup.write_text(f'#!/bin/sh\necho {tmp_path / "missing/cargo"}\n')
    assert gate.rustup_toolchain_bin(root=root, env={'PATH': ''}) is None


def test_cargo_passes_only_with_parsed_tests_and_no_failures(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    ok = 'test result: ok. 12 passed; 0 failed; 1 ignored; 0 measured\ntest result: ok. 3 passed; 0 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (0, ok)])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'pass' and result['passed'] == 15 and result['binaries'] == 2
    # Exit 0 but nothing ran (e.g. a wrapper that masked a missing lock): failed.
    _fake(monkeypatch, [(0, BUILD_OK), (0, 'fd: unbound variable\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and 'no test summaries' in result['error']
    bad = ok + 'test a::b ... FAILED\ntest result: FAILED. 4 passed; 1 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (101, bad)])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['a::b']
    _fake(monkeypatch, [(1, 'error[E0425]: cannot find value\nerror: could not compile\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and 'E0425' in result['error']


@pytest.mark.parametrize('build', ['', 'error: compile failed\n',
                                   '{"reason":"build-finished","success":false}\n',
                                   BUILD_OK + '{"reason":"compiler-message","message":{"level":"error"}}\n'])
def test_masked_compile_failure_never_runs_stale_test_binaries(tmp_path, monkeypatch, build):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, build), (0, 'test result: ok. 8 passed; 0 failed; 0 ignored\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and not result['build_complete']
    assert len(calls) == 1 and '--message-format=json-render-diagnostics' in calls[0]
    calls = _fake(monkeypatch, [(0, build), (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    assert gate.benches(ROOT, tmp_path, ARGS, {})['status'] == 'failed'
    assert len(calls) == len(gate.BENCHES)


def test_zero_exit_does_not_mask_test_failure_counts(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    _fake(monkeypatch, [(0, BUILD_OK), (0, 'test a::b ... FAILED\n'
                                             'test result: FAILED. 8 passed; 1 failed; 0 ignored\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['a::b']


@pytest.mark.parametrize('problem', ['timeout', 'missing-summary', 'moved', 'base-empty', 'shared-test'])
def test_base_comparison_does_not_mask_infrastructure_failures(tmp_path, monkeypatch, problem):
    suite = dict(suite='cargo', status='pass', passed=8, failed=0, failing=[], exit=0)
    head = dict(head='a', suites=[dict(suite)], status='pass', dirty=False, moved=False)
    base = dict(head='b', suites=[dict(suite)], status='pass', dirty=False, moved=False)
    if problem == 'timeout':
        head['suites'][0].update(status='failed', exit=124)
    elif problem == 'missing-summary':
        head['suites'][0].update(status='failed')
    elif problem == 'moved':
        head['moved'] = True
    elif problem == 'base-empty':
        base['suites'][0].update(status='failed', passed=0)
    else:
        for side in (head, base):
            side['suites'][0].update(status='failed', failed=1, failing=['a::b'], exit=101)
    monkeypatch.setattr(gate, 'gate', lambda source, *a, **k: head if source == ROOT else base)
    monkeypatch.setattr(gate, 'checked', lambda *a, **k: '')
    args = SimpleNamespace(task='test', source=ROOT, base='base', skip_live_planners=False)
    code = gate.run_gate(args, tmp_path)
    assert code == (0 if problem == 'shared-test' else 1)


def test_child_shell_preserves_errexit_inside_agent_and_list(tmp_path):
    script = tmp_path / 'gate.sh'
    script.write_text('set -euo pipefail\nfalse\nprintf false-green\\n\n')
    result = subprocess.run(['bash', '-c', 'true && eval \'bash "$1"\' && echo green', '_', str(script)],
                            text=True, capture_output=True)
    assert result.returncode != 0 and not result.stdout


def test_live_planner_skip_is_explicit_and_recorded(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**{**vars(ARGS), 'skip_live_planners': True}), {})
    test_cmd = calls[1]
    assert test_cmd[-1] == '--exact' and test_cmd.count('--skip') == len(set(gate.FULL_TESTS) | set(gate.LIVE_PLANNERS) | set(gate.MERGE_INSTALLED_PLANNERS))
    assert result['skipped_live_planners'] == list((*gate.LIVE_PLANNERS, *gate.INSTALLED_PLANNERS, *gate.MERGE_INSTALLED_PLANNERS))
    record = {'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [result]}
    assert 'live planner tests skipped' in gate.render(record)


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('nextest', [False, True])
def test_default_and_full_select_the_same_named_tests_in_both_runners(tmp_path, monkeypatch, full, nextest):
    (tmp_path / 'logs').mkdir()
    ok = 'test result: ok. 1 passed; 0 failed; 0 ignored\n'
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, ok), (0, ok)])
    if nextest:
        monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest')
        monkeypatch.setattr(gate, 'parse_nextest', lambda report: ([(1, 0, 0)], []))
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**vars(ARGS), full=full), {})
    assert result['status'] == 'pass' and result['full'] == full
    assert result['skipped_full_tests'] == ([] if full else list(gate.FULL_TESTS))
    assert result['skipped_live_planners'] == []
    command = calls[1]
    if nextest:
        assert command[command.index('--profile') + 1] == ('cpu-gate-full' if full else 'cpu-gate')
        assert ('-E' in command) != full
        if not full:
            assert command[command.index('-E') + 1] == ' & '.join(f'not test(={name})' for name in gate.FULL_TESTS)
    else:
        assert command.count('--skip') == (0 if full else len(gate.FULL_TESTS))
        if not full:
            assert command[-1] == '--exact'
            assert command[command.index('--') + 1:-1:2] == ['--skip'] * len(gate.FULL_TESTS)
            assert command[command.index('--') + 2:-1:2] == list(gate.FULL_TESTS)
    record = {'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [result]}
    assert ('merge gate (use --full for releases)' in gate.render(record)) != full
    assert len(gate.FULL_TESTS) == len(set(gate.FULL_TESTS))
    for name in gate.INSTALLED_PLANNERS:
        assert name in gate.FULL_TESTS
    assert not set(gate.MERGE_INSTALLED_PLANNERS).intersection(gate.FULL_TESTS)


@pytest.mark.parametrize('nextest', [False, True])
@pytest.mark.parametrize('installed', [False, True])
def test_full_gate_runs_installed_planners_without_inventory_prechecks(tmp_path, monkeypatch, nextest, installed):
    (tmp_path / 'logs').mkdir()
    hf_home, programs = tmp_path / 'hf', tmp_path / 'PROGRAMS.json'
    if installed:
        hf_home.mkdir()
        programs.write_text('{"programs":[]}')
    summary = ('test result: ok. 3 passed; 0 failed; 0 ignored\n' if installed else
               ''.join(f'test {name} ... FAILED\nrequired input missing\n' for name in gate.INSTALLED_PLANNERS)
               + 'test result: FAILED. 0 passed; 3 failed; 0 ignored\n')
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0 if installed else 101, summary),
                              (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    if nextest:
        monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest')
        monkeypatch.setattr(gate, 'parse_nextest', lambda report:
            ([(3, 0, 0)], []) if installed else ([(0, 3, 0)], list(gate.INSTALLED_PLANNERS)))
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**vars(ARGS), full=True), {
        'CUTEAFD_DEPLOYMENT_TEST_HF_HOME': str(hf_home), 'CUTEAFD_DEPLOYMENT_TEST_PROGRAMS': str(programs)})
    assert result['status'] == ('pass' if installed else 'failed')
    assert result['skipped_full_tests'] == result['skipped_live_planners'] == []
    assert set(result['failing']) == (set() if installed else set(gate.INSTALLED_PLANNERS))
    assert '--skip' not in calls[1] and '-E' not in calls[1]
    receipt = gate.render({'task': 't', 'verdict': result['status'], 'head': 'a' * 40,
                           'dirty': False, 'suites': [result]})
    assert 'planner tests skipped' not in receipt


@pytest.mark.parametrize('nextest', [False, True])
def test_failed_live_v4_provenance_is_not_reported_as_skipped(tmp_path, monkeypatch, nextest):
    (tmp_path / 'logs').mkdir()
    reason = ('stale V4 placement inputs: exported fork "125a8882" differs from tree '
              '"78cfee4e"; regenerate inputs')
    failed = ''.join(f'test {name} ... FAILED\n{reason}\n' for name in gate.LIVE_PLANNERS)
    failed += 'test result: FAILED. 8 passed; 4 failed; 0 ignored\n'
    calls = _fake(monkeypatch, [(0, BUILD_OK), (101, failed),
                                (0, 'test result: ok. 2 passed; 0 failed; 0 ignored\n')])
    if nextest:
        monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest')
        monkeypatch.setattr(gate, 'parse_nextest', lambda report: ([(8, 4, 0)], list(gate.LIVE_PLANNERS)))
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**vars(ARGS), full=True), {})
    assert result['status'] == 'failed' and result['failed'] == 4
    assert set(result['failing']) == set(gate.LIVE_PLANNERS)
    assert result['skipped_full_tests'] == result['skipped_live_planners'] == []
    assert result['error'] == '4 failed; ' + reason
    record = {'task': 't', 'verdict': 'failed', 'head': 'a' * 40, 'dirty': False, 'suites': [result]}
    receipt = gate.render(record)
    assert '| cargo | failed |' in receipt and reason in receipt
    assert 'live planner tests skipped' not in receipt
    assert 'default tier' not in receipt and 'full tier' in receipt
    assert 'installed-checkpoint planner tests skipped' not in receipt
    for name in gate.LIVE_PLANNERS:
        assert name not in ' '.join(calls[1])


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('inventory', ['home', 'hf-home', 'override'])
@pytest.mark.parametrize('programs_override', [False, True])
def test_installed_deployment_inventory_is_enabled_for_both_tiers(tmp_path, monkeypatch, full, inventory, programs_override):
    (tmp_path / 'logs').mkdir()
    calls = []
    def run_logged(cmd, path, env, cwd, timeout):
        calls.append(dict(env))
        Path(path).write_text(BUILD_OK if len(calls) == 1 else
                              'test result: ok. 1 passed; 0 failed; 0 ignored\n')
        return 0, 1.
    monkeypatch.setattr(gate, 'compiler_cache_env', lambda source, root, env: dict(env))
    monkeypatch.setattr(gate, 'nextest_for', lambda env: None)
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/gate-venv/bin/python'))
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    monkeypatch.setattr(gate, 'validate_full_test_ids', lambda *a: {})
    monkeypatch.setattr(gate, 'checked', lambda cmd, **kwargs: '')
    env = {'HOME': str(tmp_path)}
    expected = str(tmp_path / '.cache/huggingface')
    if inventory in ('hf-home', 'override'):
        env['HF_HOME'] = expected = str(tmp_path / 'hf-home')
    if inventory == 'override':
        env['CUTEAFD_DEPLOYMENT_TEST_HF_HOME'] = expected = str(tmp_path / 'selected-inventory')
    if programs_override:
        env['CUTEAFD_DEPLOYMENT_TEST_PROGRAMS'] = str(tmp_path / 'PROGRAMS.json')
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**vars(ARGS), full=full), env)
    assert result['status'] == 'pass'
    assert calls[1].get('CUTEAFD_DEPLOYMENT_TEST_HF_HOME') == expected
    manifest = env.get('CUTEAFD_DEPLOYMENT_TEST_PROGRAMS',
                       str(tmp_path / '.cache/cuteafd/builds/placement-inputs/current/share/PROGRAMS.json'))
    assert all(call['CUTEAFD_DEPLOYMENT_TEST_PROGRAMS'] == manifest for call in calls)
    assert result['deployment_test_programs'] == manifest
    assert calls[1].get('CUTEAFD_TEST_PYTHON') == '/gate-venv/bin/python'
    assert 'CUTEAFD_TEST_PYTHON' not in env
    assert ('CUTEAFD_DEPLOYMENT_TEST_HF_HOME' in env) == (inventory == 'override')


def test_full_tier_adds_only_its_missing_profile_for_an_older_base(tmp_path, monkeypatch):
    source = tmp_path / 'base'
    profile = source / 'rust/.config/nextest.toml'
    profile.parent.mkdir(parents=True)
    profile.write_text('[profile.cpu-gate]\nretries = 0\n[profile.cpu-gate.junit]\npath = "junit.xml"\n')
    (tmp_path / 'logs').mkdir()
    ok = 'test result: ok. 1 passed; 0 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (0, ok), (0, ok)])
    monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest')
    monkeypatch.setattr(gate, 'parse_nextest', lambda report: ([(1, 0, 0)], []))
    result = gate.cargo(source, tmp_path, SimpleNamespace(**vars(ARGS), full=True), {})
    assert result['status'] == 'pass'
    import tomllib
    config = tomllib.loads((tmp_path / 'nextest.toml').read_text())
    assert set(config['profile']) == {'cpu-gate', 'cpu-gate-full'}
    assert config['profile']['cpu-gate-full']['junit']['report-skipped'] == 'all'


def test_scripts_parse_failing_ids_and_empty_runs(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/usr/bin/python3'))
    _fake(monkeypatch, [(1, 'FAILED scripts/tests/test_x.py::test_y - assert 0\n=== 1 failed, 20 passed, 2 skipped in 3.10s ===\n')])
    result = gate.scripts(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['scripts/tests/test_x.py::test_y']
    assert (result['passed'], result['failed'], result['skipped']) == (20, 1, 2)
    _fake(monkeypatch, [(0, '808 passed in 100.00s\n')])
    assert gate.scripts(ROOT, tmp_path, ARGS, {})['status'] == 'pass'
    _fake(monkeypatch, [(5, 'no tests ran in 0.01s\n')])
    assert gate.scripts(ROOT, tmp_path, ARGS, {})['status'] == 'failed'


def test_build_slot_refuses_a_missing_lock_directory(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'true'], capture_output=True, text=True,
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path / 'absent')})
    assert result.returncode == 2 and 'lock directory' in result.stderr and 'missing' in result.stderr
    (tmp_path / 'locks').mkdir()
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'true'], capture_output=True, text=True,
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path / 'locks')})
    assert result.returncode == 0 and (tmp_path / 'locks/build.lock').exists()


def test_benches_record_medians_and_fail_on_a_bound_or_an_empty_run(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    ran = ('C16 block 7 + copy 32, medians: round 37.1 us, plan 9.7 us, shape 12.6 us, observe 14.8 us\n'
           'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n')
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, ran)])
    envs = []
    original = gate.run_logged
    def run_logged(cmd, log_path, env, cwd, timeout):
        envs.append(env)
        return original(cmd, log_path, env, cwd, timeout)
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    result = gate.benches(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'pass' and result['passed'] == 1
    assert result['medians']['draft_history_bench'] == {'round': 37.1, 'plan': 9.7, 'shape': 12.6, 'observe': 14.8}
    assert '--release' in calls[0] and '--ignored' in calls[1]
    assert envs[1]['CUTEAFD_POLICY_BENCH_MAX_US'] == '1200'
    record = {'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [result]}
    assert 'round 37 us' in gate.render(record)
    over = 'medians: round 1500.0 us, plan 500.0 us, shape 500.0 us, observe 500.0 us\nround median 1500.0 us exceeds 1200 us\n' \
           'test result: FAILED. 0 passed; 1 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (101, over)])
    result = gate.benches(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['draft_history_bench']
    assert result['medians']['draft_history_bench']['round'] == 1500.
    _fake(monkeypatch, [(0, BUILD_OK), (0, 'running 0 tests\n')])
    assert gate.benches(ROOT, tmp_path, ARGS, {})['status'] == 'failed'


def test_benches_flag_runs_only_the_bench_suite(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(gate, 'prepare', lambda source, root, owned=False: None)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: 'abc' if 'rev-parse' in cmd else '')
    for name in ('cargo', 'scripts', 'benches'):
        monkeypatch.setattr(gate, name, lambda s, r, a, e, n=name: seen.append(n) or {'suite': n, 'status': 'pass'})
    args = SimpleNamespace(**{**vars(ARGS), 'cargo': False, 'scripts': False, 'benches': True})
    assert gate.gate(ROOT, tmp_path, args)['status'] == 'pass' and seen == ['benches']


def _git(cwd, *args):
    subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True,
                   env={**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't',
                        'GIT_COMMITTER_EMAIL': 't@t', 'GIT_ALLOW_PROTOCOL': 'file'})


def test_pin_changes_refuses_an_uncommitted_submodule_bump(tmp_path):

    sub = tmp_path / 'sub'
    sub.mkdir()
    _git(sub, 'init', '-q')
    (sub / 'f').write_text('1')
    _git(sub, 'add', 'f')
    _git(sub, 'commit', '-qm', 'one')
    top = tmp_path / 'top'
    top.mkdir()
    _git(top, 'init', '-q')
    _git(top, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(sub), 'third_party/sub')
    _git(top, 'commit', '-qm', 'pin')
    assert gate.pin_changes(top) == []
    # Edits inside the submodule with the same commit are not a pin change.
    (top / 'third_party/sub/f').write_text('dirty')
    assert gate.pin_changes(top) == []
    _git(top / 'third_party/sub', 'checkout', '-q', 'f')
    # A checked-out newer commit (unstaged) and a staged bump are both refused.
    (top / 'third_party/sub/f').write_text('2')
    _git(top / 'third_party/sub', 'commit', '-qam', 'two')
    assert gate.pin_changes(top) == ['third_party/sub']
    _git(top, 'add', 'third_party/sub')
    assert gate.pin_changes(top) == ['third_party/sub']
    _git(top, 'commit', '-qm', 'bump')
    assert gate.pin_changes(top) == []


def test_prepare_refuses_a_moved_pin_only_in_the_gated_source(tmp_path, monkeypatch):
    """The --base worktree is the gate's own: re-detached at each base, its submodule still
    sits at the previous base's pin, which `submodule update` then fixes. Refusing there broke
    every second --base run; the gated source is still checked."""
    monkeypatch.setattr(gate, 'pin_changes', lambda source: ['third_party/sparkinfer'])
    calls = []
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: calls.append(cmd) or '')
    with pytest.raises(SystemExit, match='submodule pins differ'):
        gate.prepare(tmp_path, tmp_path / 'head')
    gate.prepare(tmp_path, tmp_path / 'base', owned=True)
    assert any('submodule' in cmd for cmd in calls)


def test_gate_records_the_starting_head_and_fails_when_it_moves(tmp_path, monkeypatch):
    """A checkout that changes while the gate runs must not be credited with its tests."""
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/gate-venv/bin/python'))
    monkeypatch.setattr(gate, 'prepare', lambda source, root, owned=False: None)
    heads = iter(['aaaa', 'bbbb'])
    def checked(cmd, **k):
        return next(heads) + '\n' if 'rev-parse' in cmd else ''
    monkeypatch.setattr(gate, 'checked', checked)
    monkeypatch.setattr(gate, 'scripts', lambda *a: {'suite': 'scripts', 'status': 'pass'})
    args = SimpleNamespace(cargo=False, scripts=True, benches=False)
    record = gate.gate(ROOT, tmp_path, args)
    assert record['head'] == 'aaaa' and record['moved'] and record['status'] == 'failed'


def test_nextest_machine_report_includes_crashes_and_skips(tmp_path):
    report = tmp_path / 'junit.xml'
    report.write_text('''<testsuites><testsuite name="crate">
      <testcase name="ok"/><testcase name="bad"><failure/></testcase>
      <testcase name="crash"><error/></testcase><testcase name="ignored"><skipped/></testcase>
    </testsuite></testsuites>''')
    assert gate.parse_nextest(report) == ([(1, 2, 1)], ['bad', 'crash'])
    assert gate.parse_nextest(tmp_path / 'missing') == ([], [])
    report.write_text('<testsuites>')
    assert gate.parse_nextest(report) == ([], [])


def test_worker_budget_reaches_both_runners(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    envs = []
    run_logged = gate.run_logged
    def capture_env(cmd, path, env, cwd, timeout):
        envs.append(env)
        return run_logged(cmd, path, env, cwd, timeout)
    monkeypatch.setattr(gate, 'run_logged', capture_env)
    gate.cargo(ROOT, tmp_path, ARGS, {})
    assert all(env['CARGO_BUILD_JOBS'] == env['RUST_TEST_THREADS'] == '4' for env in envs)
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/usr/bin/python3'))
    calls = _fake(monkeypatch, [(0, '1 passed in 1.00s\n')])
    gate.scripts(ROOT, tmp_path, ARGS, {})
    assert calls[0][calls[0].index('-n', 3) + 1] == '4'


def test_slow_runs_only_explicit_cpu_tests(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    ok = 'test result: ok. 1 passed; 0 failed; 0 ignored\n'
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, ok), *[(0, ok)] * len(gate.SLOW_TESTS)])
    args = SimpleNamespace(**vars(ARGS), slow=True)
    result = gate.cargo(ROOT, tmp_path, args, {})
    assert result['status'] == 'pass'
    assert len(calls) == 2 + len(gate.SLOW_TESTS)
    assert all('--ignored' in call and '--include-ignored' not in call for call in calls[2:])


def test_suites_overlap_without_changing_receipt_order(tmp_path, monkeypatch):
    import threading
    barrier = threading.Barrier(2, timeout=5)
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/gate-venv/bin/python'))
    monkeypatch.setattr(gate, 'prepare', lambda *a, **k: None)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: 'abc' if 'rev-parse' in cmd else '')
    def run(name):
        barrier.wait()
        return {'suite': name, 'status': 'pass'}
    monkeypatch.setattr(gate, 'cargo', lambda *a: run('cargo'))
    monkeypatch.setattr(gate, 'scripts', lambda *a: run('scripts'))
    args = SimpleNamespace(cargo=True, scripts=True, benches=False)
    assert [s['suite'] for s in gate.gate(ROOT, tmp_path, args)['suites']] == ['cargo', 'scripts']


def test_missing_nextest_report_cannot_be_masked_by_passing_doctests(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, BUILD_OK), (0, 'test result: ok. 2 passed; 0 failed; 0 ignored\n')])
    monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest')
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert calls[1][4:6] == ['nextest', 'run']
    config = tmp_path / 'nextest.toml'
    assert str(tmp_path / 'logs/nextest') in config.read_text()
    assert calls[1][calls[1].index('--config-file') + 1] == str(config)
    assert result['status'] == 'failed' and result['exit'] != 0


def test_launcher_inputs_link_readonly_files_but_copy_mutable_tree_files(tmp_path, monkeypatch):
    import launcher_sandbox
    source, cache, clone = (tmp_path / name for name in ('source', 'cache', 'clone'))
    source.mkdir()
    cache.mkdir()
    (source / 'readonly.py').write_text('original')
    (source / 'mutable.sh').write_text('original')
    with monkeypatch.context() as patch:
        patch.setattr(launcher_sandbox, '_ROOT', cache)
        patch.setattr(launcher_sandbox, '_TEMPLATES', {})
        launcher_sandbox.copy_tree(source, clone)
        other = tmp_path / 'other'
        launcher_sandbox.copy_tree(source, other)
        assert (clone / 'readonly.py').stat().st_ino == (other / 'readonly.py').stat().st_ino
        (clone / 'mutable.sh').write_text('edited')
        assert (other / 'mutable.sh').read_text() == (source / 'mutable.sh').read_text() == 'original'
        launcher_sandbox.copy_input(source / 'readonly.py', tmp_path / 'input')
        assert (tmp_path / 'input').stat().st_ino != (source / 'readonly.py').stat().st_ino


def test_scratch_report_groups_by_test_and_stem(tmp_path):
    tmp = tmp_path / 'tmp'
    for worker, repeat in (('popen-gw0', 0), ('popen-gw3', 1)):
        path = tmp / 'pytest' / worker / f'test_big_fixture{repeat}' / 'blob'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'x' * (1 << 20))
    (tmp / 'pytest/popen-gw1/test_small0').mkdir(parents=True)
    (tmp / 'pytest/popen-gw1/test_small0/a').write_bytes(b'x' * 4096)
    for pid in (11, 12):
        (tmp / f'cuteafd-fabric-{pid}').mkdir(parents=True)
        (tmp / f'cuteafd-fabric-{pid}/f').write_bytes(b'x' * 65536)
    (tmp / 'tmpab12cd34').write_bytes(b'x' * 8192)
    rows = gate.scratch_usage(tmp, top=3)
    assert [row['name'] for row in rows] == ['pytest:test_big_fixture', 'cuteafd-fabric', 'tmp*']
    assert rows[0]['bytes'] >= 2 << 20
    assert gate.scratch_usage(tmp_path / 'missing') == []
    text = gate.render({'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [], 'scratch': rows})
    assert 'Largest scratch consumers' in text and 'pytest:test_big_fixture' in text


def _scratch_tree(root):
    for side in ('head', 'base'):
        for name in gate.SCRATCH:
            (root / side / name / 'deep').mkdir(parents=True)
            (root / side / name / 'deep/file').write_text('x')
        (root / side / 'logs').mkdir()
        (root / side / 'logs/pytest.log').write_text('kept')
    (root / 'head/tmp/deep').chmod(0o500)  # tests leave read-only directories behind


@pytest.mark.parametrize('keep, outcome', [(False, 'pass'), (False, 'raise'), (True, 'pass')])
def test_scratch_is_removed_on_every_exit_unless_kept(tmp_path, monkeypatch, keep, outcome):
    root = tmp_path / 'builds/task'
    _scratch_tree(root)
    monkeypatch.setattr(gate, 'BUILDS', tmp_path / 'builds')
    monkeypatch.setattr(gate.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(gate, 'gate_lease', lambda: gate.nullcontext(None))

    def run_gate(args, root):
        if outcome == 'raise':
            raise KeyboardInterrupt('signal 15')
        return 0
    monkeypatch.setattr(gate, 'run_gate', run_gate)
    monkeypatch.setattr(sys, 'argv', ['cpu-gate', '--task', 'task'] + (['--keep'] if keep else []))
    if outcome == 'raise':
        assert gate.main() in (130, 143)
        assert json.loads((root / 'cpu-gate.json').read_text())['verdict'] == 'terminated'
    else:
        assert gate.main() == 0
    for side in ('head', 'base'):
        assert (root / side / 'logs/pytest.log').read_text() == 'kept'
        for name in gate.SCRATCH:
            assert (root / side / name).exists() == keep
    if keep:
        (root / 'head/tmp/deep').chmod(0o700)  # let pytest's basetemp cleanup remove it


def test_interrupt_stops_running_suites(tmp_path):
    log_path = tmp_path / 'log'
    import threading
    result = {}
    worker = threading.Thread(target=lambda: result.update(code=gate.run_logged(
        ['sh', '-c', 'sleep 60 & wait'], log_path, dict(os.environ), tmp_path, 120)))
    gate.STOPPING.clear()
    worker.start()
    try:
        deadline = time.time() + 10
        while not gate.CHILDREN and time.time() < deadline:
            time.sleep(0.01)
        with pytest.raises(KeyboardInterrupt):
            gate.interrupt(15, None)
        worker.join(10)
        assert not worker.is_alive() and result['code'][0] != 0
        assert gate.run_logged(['true'], log_path, dict(os.environ), tmp_path, 10)[0] == 130
    finally:
        gate.STOPPING.clear()


def test_gate_keeps_cargo_cache_and_scrubs_only_script_fixtures(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    inherited = {
        'CMAKE_C_COMPILER_LAUNCHER': '/tmp/kache',
        'CMAKE_CXX_COMPILER_LAUNCHER': '/tmp/kache',
        'CMAKE_CUDA_COMPILER_LAUNCHER': '/tmp/sccache',
        'CUTEAFD_KACHE': '/tmp/kache',
        'CUTEAFD_KACHE_MODE': 'enabled',
        'KACHE_DIR': '/tmp/kache-dir',
        'CUTEAFD_SCCACHE_DIR': '/tmp/sccache-dir',
        'SCCACHE_DIR': '/tmp/sccache',
        'RUSTC_WRAPPER': '/tmp/kache',
        'RUSTC_WORKSPACE_WRAPPER': '/tmp/kache',
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('KEEP_PLAIN_VAR', 'kept')
    seen_envs = []

    def run_logged(cmd, log_path, env, cwd, timeout):
        seen_envs.append(dict(env))
        if 'pytest' in cmd:
            Path(log_path).write_text('=== 5 passed in 0.10s ===\n')
        elif '--no-run' in cmd:
            Path(log_path).write_text(BUILD_OK)
        else:
            Path(log_path).write_text('test result: ok. 3 passed; 0 failed; 0 ignored\n')
        return 0, 1.0

    monkeypatch.setattr(gate, 'prepare', lambda source, root, owned=False: None)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: 'deadbeef' if 'rev-parse' in cmd else '')
    monkeypatch.setattr(gate, 'compiler_cache_env', lambda source, root, env: dict(env))
    monkeypatch.setattr(gate, 'nextest_for', lambda env: None)
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/usr/bin/python3'))
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    monkeypatch.setattr(gate, 'validate_full_test_ids', lambda *a: {})
    args = SimpleNamespace(**vars(ARGS), cargo=True, scripts=True, benches=False, jobs=1)
    record = gate.gate(ROOT, tmp_path, args)
    assert record['status'] == 'pass'
    expected_scrubbed = sorted(inherited)
    assert record['scrubbed_env'] == []
    assert record['suites'][0]['scrubbed_env'] == []
    assert record['suites'][1]['scrubbed_env'] == expected_scrubbed
    assert len(seen_envs) == 3
    for env in seen_envs:
        assert env.get('KEEP_PLAIN_VAR') == 'kept'
        for key in expected_scrubbed:
            if 'CARGO_TARGET_DIR' in env:
                assert env[key] == inherited[key]
            else:
                assert key not in env


def test_v4_live_planners_are_explicit_full_tier_requirements():
    assert set(gate.LIVE_PLANNERS) <= set(gate.FULL_TESTS)
    assert 'families::deepseek_v4::placement_tests::planner_equals_runtime_deepseek_v4_measured' in gate.LIVE_PLANNERS
    assert 'serving_capacity::v4_placement::tp2_tests::real_tp2_costs_match_catalog_reference' in gate.LIVE_PLANNERS


@pytest.mark.parametrize('code,text,error', [
    (0, 'kept::test: test\n', None),
    (0, 'kept::test: test\nkept::test: test\n', None),
    (0, 'kept::test_suffix: test\n', 'match zero tests'),
    (0, '', 'inventory is empty'),
    (1, 'kept::test: test\n', 'inventory failed'),
    (124, 'kept::test: test\n', 'inventory failed'),
])
def test_full_exclusions_require_exact_compiled_test_ids(tmp_path, monkeypatch, code, text, error):
    (tmp_path / 'logs').mkdir()
    monkeypatch.setattr(gate, 'FULL_TESTS', ('kept::test',))
    def run_logged(cmd, path, env, cwd, timeout):
        assert '--list' in cmd and '--format' in cmd
        assert 'build-slot.sh' not in ' '.join(cmd)
        path.write_text(text)
        return code, 1.
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    result = gate.validate_full_test_ids(ROOT, tmp_path, ARGS, {})
    if error:
        assert error in result['error']
    else:
        assert 'error' not in result
    assert result['missing_full_tests'] == ([] if 'kept::test: test' in text else ['kept::test'])


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('nextest', [False, True])
def test_dead_full_exclusion_refuses_before_either_runner(tmp_path, monkeypatch, full, nextest):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, BUILD_OK)])
    monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest' if nextest else None)
    monkeypatch.setattr(gate, 'validate_full_test_ids', lambda *a: {
        'error': 'FULL_TESTS ids match zero tests: removed::test', 'missing_full_tests': ['removed::test']})
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**vars(ARGS), full=full), {})
    assert result['status'] == 'failed' and 'removed::test' in result['error']
    assert len(calls) == 1


def test_named_step5_refusal_cases_are_not_full_tier_exclusions():
    assert not any('local_without_bf16_tail_export' in name or 'spark_count_one' in name
                   or 'spark_count_three' in name or 'spark_count_eight' in name
                   for name in gate.FULL_TESTS)
    assert 'plan::families::step5::tests::native_prefill_refuses_missing_spark_packages' in gate.FULL_TESTS
    assert 'plan::families::step5::tests::nvfp4_min_card_admits_native_w4a4_and_bf16_root_stays_reference_only' in gate.FULL_TESTS


def test_gate_requirements_check_prefers_gate_venv_and_fails_fast_on_mismatch(tmp_path, monkeypatch):
    expected = gate.parse_requirements()
    assert len(expected) >= 40 and expected.get('pytest') == '9.1.1'
    assert 'torch' not in expected and 'transformers' not in expected
    main_repo = tmp_path / 'main'
    worktree = tmp_path / 'worktree'
    gate_venv = tmp_path / '.cache/cuteafd/gate-venv/bin'
    for path in (main_repo / '.venv/bin', worktree, gate_venv):
        path.mkdir(parents=True)
    for python in (main_repo / '.venv/bin/python', gate_venv / 'python'):
        python.write_text('#!/bin/sh\n')
        python.chmod(0o755)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(gate, 'REPO', main_repo)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: str(main_repo / '.git') + '\n')
    monkeypatch.setattr(gate, 'installed_packages', lambda python: dict(expected))
    # Worktree without .venv linked prefers ~/.cache/cuteafd/gate-venv over main checkout's .venv.
    assert gate.python_for(worktree) == gate_venv / 'python'
    # Worktree with .venv linked uses its own .venv first.
    (worktree / '.venv/bin').mkdir(parents=True)
    worktree_py = worktree / '.venv/bin/python'
    worktree_py.write_text('#!/bin/sh\n')
    worktree_py.chmod(0o755)
    assert gate.python_for(worktree) == worktree_py
    # Mismatched venv fails fast with exact sync command: missing, wrong version, or extra quant packages.
    for mutated, detail in (
        ({k: v for k, v in expected.items() if k != 'pytest-xdist'}, 'missing: pytest-xdist==3.8.0'),
        ({**expected, 'numpy': '1.26.4'}, 'version mismatch: numpy==1.26.4 (expected 2.5.3)'),
        ({**expected, 'torch': '2.14.1+cu130', 'transformers': '5.19.0'}, 'unexpected packages: torch==2.14.1+cu130'),
    ):
        monkeypatch.setattr(gate, 'installed_packages', lambda python, m=mutated: m)
        with pytest.raises(SystemExit) as refused:
            gate.python_for(worktree)
        message = str(refused.value)
        assert detail in message and 'uv pip sync --python' in message


@pytest.mark.parametrize('nextest', [False, True])
def test_missing_installed_programs_manifest_surfaces_in_receipt_error(tmp_path, monkeypatch, nextest):
    (tmp_path / 'logs').mkdir()
    test_id = 'shared::experts::rtx::local_physical_tests::installed_glm_local_physical_equals_serving'
    reason = f'required installed PROGRAMS.json missing: {tmp_path}/placement-inputs/current/share/PROGRAMS.json'
    failed = f'test {test_id} ... FAILED\nError: {reason}\ntest result: FAILED. 8 passed; 1 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (101, failed),
                        (0, 'test result: ok. 2 passed; 0 failed; 0 ignored\n')])
    if nextest:
        monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest')
        monkeypatch.setattr(gate, 'parse_nextest', lambda report: ([(8, 1, 0)], [test_id]))
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**vars(ARGS), full=False), {})
    assert result['status'] == 'failed' and result['failing'] == [test_id]
    assert result['error'] == '1 failed; ' + reason
    receipt = gate.render({'task': 't', 'verdict': 'failed', 'head': 'a' * 40, 'dirty': False, 'suites': [result]})
    assert reason in receipt


def test_help_warns_against_systemd_supplementary_groups():
    assert '216/GROUP' in gate.__doc__
    assert 'gate-requirements.txt' in gate.__doc__


def test_gate_lease_counts_waiters_and_releases(tmp_path):
    with gate.gate_lease(tmp_path, slots=2, timeout=0) as first:
        with gate.gate_lease(tmp_path, slots=2, timeout=0) as second:
            assert (first['slot'], second['slot']) == (0, 1)
            with pytest.raises(TimeoutError, match='no gate lease free'):
                with gate.gate_lease(tmp_path, slots=2, timeout=0):
                    pytest.fail('a third gate acquired a lease')
        with gate.gate_lease(tmp_path, slots=2, timeout=0) as reused:
            assert reused['slot'] == 1


def _queued_waiter(tmp_path, slots, timeout=30):
    """A gate_lease waiter in a child process; prints 'acquired SLOT' and holds until stdin closes."""
    code = (f'import importlib.machinery, importlib.util, sys\n'
            f'from pathlib import Path\n'
            f'loader = importlib.machinery.SourceFileLoader("cpu_gate", {str(ROOT / "scripts/dev/cpu-gate")!r})\n'
            f'spec = importlib.util.spec_from_loader("cpu_gate", loader)\n'
            f'gate = importlib.util.module_from_spec(spec); loader.exec_module(gate)\n'
            f'gate.GATE_LEASE_POLL = 0.05\n'
            f'with gate.gate_lease(Path({str(tmp_path)!r}), slots={slots}, timeout={timeout}) as lease:\n'
            f'    print("acquired", lease["slot"], flush=True)\n'
            f'    sys.stdin.read()\n')
    return subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _tickets(tmp_path):
    return sorted(path.name for path in (tmp_path / 'cpu-gate-q').glob('*.ticket'))


def _wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'condition not reached'
        time.sleep(0.02)


def _release(process):
    process.stdin.close()
    process.wait(timeout=10)


def test_gate_lease_honours_arrival_order(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'GATE_LEASE_POLL', 0.05)
    waiters = []
    try:
        with gate.gate_lease(tmp_path, slots=1, timeout=0):
            for _ in range(2):
                waiters.append(_queued_waiter(tmp_path, slots=1))
                _wait_for(lambda: len(_tickets(tmp_path)) == len(waiters))
            early, late = waiters
            # The later waiter sees the free slot too, but the earlier ticket is ahead of it.
        assert early.stdout.readline().split() == ['acquired', '0']
        time.sleep(0.3)
        assert late.poll() is None and len(_tickets(tmp_path)) == 1
        _release(early)
        assert late.stdout.readline().split() == ['acquired', '0']
        _wait_for(lambda: not _tickets(tmp_path))
        _release(late)
    finally:
        for process in waiters:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_gate_lease_queue_position_blocks_a_newer_waiter_when_a_slot_frees(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'GATE_LEASE_POLL', 0.05)
    queue = tmp_path / 'cpu-gate-q'
    queue.mkdir()
    hw = gate.HWLOCK
    # An older live waiter (flocked ticket) is ahead, so a free slot is not ours to take.
    older, handle = hw._publish(queue, hw._next_seq(queue), 'ticket', dict(pid=os.getpid(), created=time.time()))
    try:
        with pytest.raises(TimeoutError):
            with gate.gate_lease(tmp_path, slots=1, timeout=0):
                pytest.fail('jumped an older live ticket')
        assert _tickets(tmp_path) == [older.name]
    finally:
        handle.close()
    older.unlink()
    with gate.gate_lease(tmp_path, slots=1, timeout=0) as lease:
        assert lease['slot'] == 0


def test_gate_lease_skips_and_removes_dead_tickets(tmp_path):
    queue = tmp_path / 'cpu-gate-q'
    queue.mkdir()
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    stale = queue / f'{1:012d}.{dead.pid}.ticket'
    stale.write_text(json.dumps(dict(pid=dead.pid, created=time.time() - 60)))
    (queue / 'seq').write_text('1')
    with gate.gate_lease(tmp_path, slots=1, timeout=0) as lease:
        assert lease['slot'] == 0 and lease['ticket'] == 2
        assert not stale.exists() and not _tickets(tmp_path)


def test_gate_lease_timeout_removes_its_ticket(tmp_path):
    with gate.gate_lease(tmp_path, slots=1, timeout=0):
        with pytest.raises(TimeoutError, match='no gate lease free'):
            with gate.gate_lease(tmp_path, slots=1, timeout=0):
                pytest.fail('a second gate acquired the only lease')
        assert not _tickets(tmp_path)
    assert not _tickets(tmp_path)


def test_gate_lease_stopping_removes_its_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'STOPPING', gate.threading.Event())
    gate.STOPPING.set()
    with gate.gate_lease(tmp_path, slots=1, timeout=60):
        with pytest.raises(KeyboardInterrupt, match='terminated while waiting'):
            with gate.gate_lease(tmp_path, slots=1, timeout=60):
                pytest.fail('a stopped gate acquired a lease')
        assert not _tickets(tmp_path)


def test_gate_lease_admits_the_oldest_n_waiters(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'GATE_LEASE_POLL', 0.05)
    waiters = []
    try:
        with gate.gate_lease(tmp_path, slots=2, timeout=0), gate.gate_lease(tmp_path, slots=2, timeout=0):
            for _ in range(3):
                waiters.append(_queued_waiter(tmp_path, slots=2))
                _wait_for(lambda: len(_tickets(tmp_path)) == len(waiters))
        first, second, third = waiters
        first_slot, second_slot = (process.stdout.readline().split()[1] for process in (first, second))
        assert {first_slot, second_slot} == {'0', '1'}
        time.sleep(0.3)
        assert third.poll() is None and len(_tickets(tmp_path)) == 1
        _release(second)
        assert third.stdout.readline().split() == ['acquired', second_slot]
        _wait_for(lambda: not _tickets(tmp_path))
        _release(first)
        _release(third)
    finally:
        for process in waiters:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_compiler_cache_setup_retains_cache_configuration(tmp_path):
    wrapper = tmp_path / 'kache'
    wrapper.write_text('#!/bin/sh\n[ "$1" = --version ] && exit 0\nexit 1\n')
    wrapper.chmod(0o755)
    env, _ = gate.scrub_compiler_cache_env(os.environ)
    env.update(HOME=str(tmp_path), CUTEAFD_KACHE=str(wrapper),
               CUTEAFD_KACHE_CACHE_DIR=str(tmp_path / 'cache'), KEEP_VAR='kept')
    configured = gate.compiler_cache_env(ROOT, tmp_path / 'build', env)
    assert configured['CUTEAFD_KACHE_ACTIVE'] == str(wrapper)
    assert configured['RUSTC_WRAPPER'] == str(ROOT / 'scripts/build/compiler-cache.sh')
    assert configured['KACHE_CACHE_DIR'] == str(tmp_path / 'cache' / gate.platform.machine())
    assert configured['KEEP_VAR'] == 'kept'
    assert configured['CARGO_INCREMENTAL'] == '0'
    assert gate.compiler_cache_receipt(configured, tmp_path / 'target')['mode'] == 'kache'
    warning = Path(configured['CUTEAFD_KACHE_WARNING_DIR']) / 'first-failure'
    warning.parent.mkdir()
    warning.write_text('exit 124: timed out')
    assert gate.compiler_cache_receipt(configured, tmp_path / 'target')['mode'] == 'fallback'


def test_missing_compiler_cache_is_fail_soft(tmp_path, capsys):
    env, _ = gate.scrub_compiler_cache_env(os.environ)
    env['CUTEAFD_KACHE'] = '/nonexistent/kache'
    configured = gate.compiler_cache_env(ROOT, tmp_path, env)
    assert not configured.get('RUSTC_WRAPPER')
    assert 'using plain compilers' in capsys.readouterr().err


def test_build_slot_termination_while_waiting_is_not_success(tmp_path):
    import fcntl
    lock = (tmp_path / 'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    process = subprocess.Popen(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'true'],
                               env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path), 'CUTEAFD_BUILD_SLOTS': '1'},
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        # SIGTERM the whole waiting group, just as systemd does on stop.
        time.sleep(0.2)
        os.killpg(process.pid, 15)
        _, error = process.communicate(timeout=5)
        assert process.returncode == 143
        assert 'terminated' in error
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()
        lock.close()


def test_build_leg_notification_records_termination(tmp_path):
    helper = ROOT / 'scripts/lib/build-supervision.sh'
    process = subprocess.Popen(['bash', '-c', f'source "{helper}"; exec 8>"$1"; work() {{ sleep 60; }}; build_run_leg fixture work',
                                'test', str(tmp_path / 'completion')],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        time.sleep(0.2)
        os.killpg(process.pid, 15)
        _, error = process.communicate(timeout=5)
        assert (tmp_path / 'completion').read_text() == 'fixture 143\n'
        assert 'terminated' in error
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


def test_build_slot_reaps_term_ignoring_descendants_in_separate_groups(tmp_path):
    script = """import subprocess, sys, signal, os
child = subprocess.Popen([sys.executable, '-c', 'import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)'], start_new_session=True)
child.wait()
"""
    process = subprocess.Popen(['bash', str(ROOT / 'scripts/build/build-slot.sh'), sys.executable, '-c', script],
                               env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)},
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    import selectors
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(5)
        pid = int(process.stdout.readline())
    try:
        process.terminate()  # only the entrypoint, not the child group
        _, error = process.communicate(timeout=10)
        assert process.returncode == 143, error
        assert not Path(f'/proc/{pid}').exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 9)


def test_cpu_gate_terminated_in_lease_queue_writes_nonzero_receipt(tmp_path):
    locks = tmp_path / '.cache/cuteafd'
    locks.mkdir(parents=True)
    with gate.gate_lease(locks, slots=2, timeout=0):
        with gate.gate_lease(locks, slots=2, timeout=0):
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/dev/cpu-gate'), '--task', 'terminated-lease',
                                        '--skip-live-planners'],
                                       env={**os.environ, 'HOME': str(tmp_path)}, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                import selectors
                selector = selectors.DefaultSelector()
                selector.register(process.stderr, selectors.EVENT_READ)
                assert selector.select(timeout=10)
                line = process.stderr.readline()
                if 'resolved child tools' in line:
                    line = process.stderr.readline()
                assert 'waiting for gate lease' in line
                process.terminate()
                process.communicate(timeout=10)
                assert process.returncode == 143
                record = json.loads((locks / 'builds/terminated-lease/cpu-gate.json').read_text())
                assert record['verdict'] == 'terminated' and record['exit'] == 143
                selector.close()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, 9)
                    process.wait()


def test_term_ignoring_suite_has_bounded_unwind_and_terminated_receipt(tmp_path, monkeypatch):
    import threading
    (tmp_path / 'logs').mkdir()
    ready = tmp_path / 'ready'
    root = tmp_path / 'builds/term-ignoring'
    root.mkdir(parents=True)
    suite_log = tmp_path / 'logs/suite.log'
    command = [sys.executable, '-c',
               'import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
               f'pathlib.Path({str(ready)!r}).touch(); time.sleep(120)']
    monkeypatch.setattr(gate, 'BUILDS', tmp_path / 'builds')
    monkeypatch.setattr(gate, 'gate_lease', lambda: gate.nullcontext(None))
    monkeypatch.setattr(gate, 'checked', lambda *a, **k: 'deadbeef')
    monkeypatch.setattr(gate.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['cpu-gate', '--task', 'term-ignoring'])
    gate.STOPPING.clear()
    gate.TERMINATED_SIGNAL = None

    def run_gate(args, root):
        with gate.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(gate.run_logged, command, suite_log, dict(os.environ), tmp_path, 180)
            deadline = time.monotonic() + 10
            while not ready.exists():
                assert time.monotonic() < deadline
                threading.Event().wait(0.01)
            gate.interrupt(15, None)
            result.result()

    monkeypatch.setattr(gate, 'run_gate', run_gate)
    try:
        started = time.monotonic()
        assert gate.main() == 143
        assert time.monotonic() - started < 13
        record = json.loads((root / 'cpu-gate.json').read_text())
        assert record['verdict'] == 'terminated' and record['exit'] == 143
        assert not gate.CHILDREN
    finally:
        gate.STOPPING.clear()
        gate.TERMINATED_SIGNAL = None


def test_child_path_resolves_rustup_before_fixture_changes_home(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    toolchain = home / '.rustup/toolchains/stable/bin'
    toolchain.mkdir(parents=True)
    for name in ('cargo', 'rustc'):
        binary = toolchain / name
        binary.write_text('#!/bin/sh\nprintf "resolved-toolchain\\n"\n')
        binary.chmod(0o755)
    rustup = home / '.cargo/bin/rustup'
    rustup.parent.mkdir(parents=True)
    rustup.write_text(f'#!/bin/sh\nprintf "%s\\n" "{toolchain / "cargo"}"\n')
    rustup.chmod(0o755)
    brew = tmp_path / 'brew/bin'
    brew.mkdir(parents=True)
    (brew / 'nasm').write_text('#!/bin/sh\nexit 0\n')
    (brew / 'nasm').chmod(0o755)
    monkeypatch.setattr(gate, 'HOMEBREW_BIN', brew)
    monkeypatch.setattr(gate, 'RUSTUP_HOME', tmp_path / 'absent')
    path, receipt = gate.resolved_child_tools({'HOME': str(home), 'PATH': str(rustup.parent)})
    result = subprocess.run(['rustc', '-V'], env={'HOME': str(tmp_path / 'fixture-home'), 'PATH': path},
                            text=True, capture_output=True, check=True)
    assert result.stdout == 'resolved-toolchain\n'
    assert receipt['tools']['rustc'] == str(toolchain / 'rustc')
    assert receipt['tools']['nasm'] == str(brew / 'nasm')


def test_cache_stats_uses_configured_build_event_root(tmp_path, monkeypatch):
    root = tmp_path / 'head'
    stats = {'summary': {'local_hits': 12, 'remote_hits': 2, 'misses': 3}}
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(stats))
    monkeypatch.setattr(gate.subprocess, 'run', run)
    receipt = gate.compiler_cache_receipt({'CUTEAFD_KACHE_ACTIVE': '/kache',
                                          'KACHE_CONFIG': str(root / 'compiler-cache/config.toml')}, root / 'target')
    assert receipt['event_root'] == str(root)
    assert receipt['hits'] == 14 and receipt['misses'] == 3
    assert calls[-1] == ['/kache', '--json', 'stats', '--root', str(root), '--full']


def test_build_channel_waits_for_remote_descendants_to_stop(tmp_path):
    # Model docker exec/SSH: the remote supervisor and exporter belong to a
    # different session and receive only stdin EOF, not the caller's signal.
    supervisor = str(ROOT / 'scripts/build/build-process.py')
    remote = [sys.executable, supervisor, '--channel', '--grace', '0.5', '--',
              sys.executable, '-c', 'import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
              'print(os.getpid(),flush=True); time.sleep(30)']
    relay = 'import subprocess,sys; sys.exit(subprocess.call(sys.argv[1:], start_new_session=True))'
    with subprocess.Popen([sys.executable, supervisor, '--pipe', '--grace', '3', '--',
                           sys.executable, '-c', relay, *remote], stdout=subprocess.PIPE, text=True) as process:
        import selectors
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(5)
            pid = int(process.stdout.readline())
        try:
            process.terminate()
            assert process.wait(timeout=5) == 143
            assert not Path(f'/proc/{pid}').exists()
        finally:
            if process.poll() is None:
                process.kill()


# Placement-input resolution: a checkout validates against the publication that records its own key.

def _publication_tree(tmp_path, edit=None):
    """A copy of this checkout's exporter sources and a store holding its publication."""
    import placement_publications as publications
    checkout = tmp_path / 'checkout'
    for relative in ('scripts/build', 'scripts/lib', 'python/tools', 'examples/configs', 'scripts/qualify/deepseek_v41'):
        shutil.copytree(ROOT / relative, checkout / relative, ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('third_party/sparkinfer.lock.json', 'benchmarks/current-matrix.json',
                     'native/cmake/shared/fp8_moe.cmake',
                     'rust/crates/cuteafd-loader/src/serving_capacity/spark_memory.json',
                     'rust/crates/cuteafd-core/src/dspark_verify.rs', 'rust/crates/cuteafd-core/src/serving_capacity.rs'):
        (checkout / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, checkout / relative)
    fingerprints = publications.checkout_fingerprints(checkout)
    home = tmp_path / 'home'
    builds = home / '.cache/cuteafd/builds'
    for kind, store in publications.STORES.items():
        for name, fingerprint in (('a' * 40, {**fingerprints[kind], 'source_tree_sha256': 'old'}),
                                  ('b' * 40, fingerprints[kind])):
            publication = builds / store / name
            (publication / 'share').mkdir(parents=True)
            (publication / 'share/PROGRAMS.json').write_text('{}')
            (publication / 'INPUTS.json').write_text(json.dumps(
                {'schema': 1, 'engine_commit': name, **fingerprint, 'exports': [], 'files': {}}, indent=2))
        (builds / store / 'current').symlink_to('a' * 40)  # current is newer dev; this branch is on an older base
    if edit:
        (checkout / edit).write_text((checkout / edit).read_text() + '\n# unpublished edit\n')
    return checkout, home


@pytest.fixture
def placement_lib(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/lib'))


def _placement_args(**overrides):
    return SimpleNamespace(**{'cargo': True, 'skip_live_planners': False, 'cargo_filter': None, 'full': True, **overrides})


def test_placement_branch_on_an_older_base_resolves_its_own_publication(tmp_path, placement_lib):
    checkout, home = _publication_tree(tmp_path)
    record, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home)}, _placement_args())
    assert error is None
    store = home / '.cache/cuteafd/builds'
    assert record['all']['path'] == str(store / 'placement-inputs' / ('b' * 40) / 'share/PROGRAMS.json')
    assert record['v4']['path'] == str(store / 'v4-placement-inputs' / ('b' * 40))
    assert record['all']['how'] == 'matching publication' and record['all']['engine_commit'] == 'b' * 40
    env = gate.apply_placement_inputs({}, record)
    assert env['CUTEAFD_DEPLOYMENT_TEST_PROGRAMS'] == record['all']['path']
    assert env['CUTEAFD_V4_PLACEMENT_INPUTS'] == record['v4']['path']
    # When current matches it is preferred.
    for kind in ('placement-inputs', 'v4-placement-inputs'):
        (store / kind / 'current').unlink()
        (store / kind / 'current').symlink_to('b' * 40)
    record, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home)}, _placement_args())
    assert error is None and record['all']['how'] == record['v4']['how'] == 'current'


def test_placement_unpublished_exporter_edit_fails_fast_before_compiling(tmp_path, placement_lib, monkeypatch):
    checkout, home = _publication_tree(tmp_path, edit='python/tools/aot/package_exl3_aot.py')
    started = time.monotonic()
    record, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home)}, _placement_args())
    assert time.monotonic() - started < 5
    assert 'no placement-inputs publication records' in error and 'python/tools/aot/package_exl3_aot.py' in error
    assert 'no v4-placement-inputs publication' in error and 'export-placement-inputs.py --output' in error
    assert record['all']['path'] is None
    # The default tier does not run live V4 planners: only the all-family inventory is required.
    _, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home)}, _placement_args(full=False))
    assert 'v4-placement-inputs' not in error and 'placement-inputs publication' in error
    # main() fails with a receipt before any lease or Cargo work.
    monkeypatch.setattr(gate, 'placement_inputs', REAL_PLACEMENT_INPUTS)
    monkeypatch.setattr(gate, 'BUILDS', tmp_path / 'builds')
    monkeypatch.setenv('HOME', str(home))
    for variable in gate.PLACEMENT_VARIABLES.values():  # a surrounding gate exports these
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(gate, 'gate_lease', lambda *a, **k: pytest.fail('took a gate lease'))
    monkeypatch.setattr(gate, 'cargo', lambda *a, **k: pytest.fail('compiled'))
    monkeypatch.setattr(sys, 'argv', ['cpu-gate', '--task', 't', '--source', str(ROOT)])
    monkeypatch.setattr(gate, '_placement_publications', lambda source: __import__('placement_publications'))
    monkeypatch.setattr(gate, 'placement_inputs', lambda source, env, args: REAL_PLACEMENT_INPUTS(checkout, env, args))
    assert gate.main() == 1
    receipt = json.loads((tmp_path / 'builds/t/cpu-gate.json').read_text())
    assert receipt['verdict'] == 'failed' and 'placement inputs: no placement-inputs publication' in receipt['error']


def test_placement_explicit_override_is_kept_only_when_it_matches(tmp_path, placement_lib):
    checkout, home = _publication_tree(tmp_path)
    store = home / '.cache/cuteafd/builds/placement-inputs'
    stale = str(store / ('a' * 40) / 'share/PROGRAMS.json')
    record, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home), 'CUTEAFD_DEPLOYMENT_TEST_PROGRAMS': stale},
                                          _placement_args(full=False))
    assert 'explicit CUTEAFD_DEPLOYMENT_TEST_PROGRAMS' in error and 'source_tree_sha256' in error
    good = str(store / ('b' * 40) / 'share/PROGRAMS.json')
    record, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home), 'CUTEAFD_DEPLOYMENT_TEST_PROGRAMS': good},
                                          _placement_args(full=False))
    assert error is None and record['all'] == dict(path=good, how='explicit CUTEAFD_DEPLOYMENT_TEST_PROGRAMS',
                                                    key=record['all']['key'], matches=True)
    assert gate.apply_placement_inputs({'CUTEAFD_DEPLOYMENT_TEST_PROGRAMS': good}, record)[
        'CUTEAFD_DEPLOYMENT_TEST_PROGRAMS'] == good


def test_placement_resolution_is_skipped_without_installed_planners(tmp_path, placement_lib):
    checkout, home = _publication_tree(tmp_path, edit='python/tools/aot/package_exl3_aot.py')
    for args in (_placement_args(skip_live_planners=True), _placement_args(cargo=False)):
        record, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home)}, args)
        assert error is None and 'skipped' in record
    record, error = REAL_PLACEMENT_INPUTS(checkout, {'HOME': str(home)}, _placement_args(cargo_filter='some_test'))
    assert error is None and record['all']['path'] is None  # focused runs: resolve when possible, never refuse


def test_placement_key_matches_the_real_checkout_and_its_current_publication():
    sys.path.insert(0, str(ROOT / 'scripts/lib'))
    import placement_publications as publications
    fingerprints = publications.checkout_fingerprints(ROOT)
    assert set(fingerprints) == {'all', 'v4'}
    assert set(fingerprints['v4']['engine_sources_sha256']) <= set(fingerprints['all']['engine_sources_sha256'])
    assert publications.inputs_key(fingerprints['all']) == publications.inputs_key(json.loads(json.dumps(fingerprints['all'])))
