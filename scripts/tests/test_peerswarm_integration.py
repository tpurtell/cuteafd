"""Tests verifying CuteAFD's PeerSwarm resource scheduler & cache integration."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_ps_resource():
    spec = importlib.util.spec_from_file_location('peerswarm_resource', REPO / 'scripts/hw/peerswarm_resource.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_peerswarm_binary_and_config_present():
    ps = _load_ps_resource()
    assert (REPO / 'peerswarm.toml').is_file()
    bin_path = Path(ps.peerswarm_bin())
    assert bin_path.is_file()
    res = subprocess.run([str(bin_path), '--version'], capture_output=True, text=True, check=True)
    assert 'peerswarm' in res.stdout


def test_canonical_lock_order():
    ps = _load_ps_resource()
    assert ps.canonical(['gpu1.lock', 'sparks', 'gpu0']) == ['sparks', 'gpu0', 'gpu1']
    assert ps.canonical(['maxq', 'kakapo', 'rhea']) == ['rhea', 'kakapo', 'maxq']
    with pytest.raises(ValueError, match='unknown hardware locks'):
        ps.canonical(['nonexistent'])


def test_hold_and_exec_use_isolated_sqlite_state(tmp_path, monkeypatch):
    ps = _load_ps_resource()
    state_dir = tmp_path / 'ps-state'
    monkeypatch.setenv('PEERSWARM_STATE_DIR', str(state_dir))
    monkeypatch.setenv('PEERSWARM_ROOT', str(REPO))

    with ps.hold(['gpu0'], wait=10, owner='pytest-hold') as grant:
        assert grant['resources'] == ['gpu0']
        assert os.environ.get('CUTEAFD_HW_LOCKS_HELD') == 'gpu0'
        assert (state_dir / 'work' / 'peerswarm.db').is_file()

        # Competing sibling subprocess waiting 1s on gpu0 must time out with exit 75
        argv = ps.exec_argv(['gpu0'], ['true'], wait=1, owner='pytest-contender')
        env = {**os.environ, 'PEERSWARM_STATE_DIR': str(state_dir), 'PEERSWARM_ROOT': str(REPO)}
        res = subprocess.run(argv, env=env, capture_output=True, text=True)
        assert res.returncode == 75

    # Nested non-gate exec inside exec must be refused immediately with exit 77
    inner = ps.exec_argv(['gpu1'], ['true'], wait=1, owner='pytest-nested-inner')
    outer = ps.exec_argv(['gpu0'], inner, wait=5, owner='pytest-nested-outer')
    res_nested = subprocess.run(
        outer,
        env={**os.environ, 'PEERSWARM_STATE_DIR': str(state_dir), 'PEERSWARM_ROOT': str(REPO)},
        capture_output=True,
        text=True,
    )
    assert res_nested.returncode == ps.NESTED_EXIT

    # After exiting the context manager, gpu0 is free immediately
    argv = ps.exec_argv(['gpu0'], ['bash', '-c', 'echo "$CUTEAFD_HW_LOCKS_HELD:$PEERSWARM_LOCKS_HELD"'], wait=5, owner='pytest-after')
    res = subprocess.run(argv, env={**os.environ, 'PEERSWARM_STATE_DIR': str(state_dir), 'PEERSWARM_ROOT': str(REPO)}, capture_output=True, text=True, check=True)
    assert res.stdout.strip() == 'gpu0:gpu0'


def test_preflight_failure_exits_78(tmp_path, monkeypatch):
    ps = _load_ps_resource()
    state_dir = tmp_path / 'ps-preflight'
    monkeypatch.setenv('PEERSWARM_STATE_DIR', str(state_dir))
    monkeypatch.setenv('PEERSWARM_ROOT', str(REPO))

    with pytest.raises(ps.PreflightFailed):
        with ps.hold(['gpu1'], wait=5, owner='pytest-pf', preflight='exit 1'):
            pass

    argv = ps.exec_argv(['gpu1'], ['true'], wait=5, preflight='exit 1', owner='pytest-pf-cli')
    res = subprocess.run(argv, env={**os.environ, 'PEERSWARM_STATE_DIR': str(state_dir), 'PEERSWARM_ROOT': str(REPO)})
    assert res.returncode == 78


def test_seed_pool_pick_balances_loaded_seed(tmp_path, monkeypatch):
    ps = _load_ps_resource()
    state_dir = tmp_path / 'ps-pool'
    monkeypatch.setenv('PEERSWARM_STATE_DIR', str(state_dir))
    monkeypatch.setenv('PEERSWARM_ROOT', str(REPO))

    with ps.hold(['kakapo'], wait=10, owner='pytest-seed'):
        picked = ps.pick('lan_seeds', prefer='kakapo', fallback=['rhea', 'moa'], probe_ssh=False)
        assert picked == 'cassowary'


def test_wip_cards_and_golden_run_and_cpu_gate_import_cleanly():
    for rel in ('scripts/bench/wip-cards.py', 'scripts/bench/golden-run.py', 'scripts/dev/cpu-gate'):
        path = REPO / rel
        res = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True)
        assert res.returncode == 0, f'{rel} --help failed: {res.stderr}'
