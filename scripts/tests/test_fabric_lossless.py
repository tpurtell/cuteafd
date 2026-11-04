"""CPU-only fabric configuration guards; no NIC or switch writes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_marks_every_rc_qp_dscp26_priority3(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'traffic.cc'
    source.write_text(r'''
#include "rdma_traffic_class.h"
static_assert(cuteafd::kRdmaTrafficClass == 104, "DSCP 26, Not-ECT");
static_assert((cuteafd::kRdmaTrafficClass >> 2) == 26 && (cuteafd::kRdmaTrafficClass >> 5) == 3, "SL 3");
int main() {}
''')
    subprocess.run([compiler, '-std=c++17', '-I', str(ROOT / 'native/shared/src'), str(source),
                    '-o', str(tmp_path / 'traffic')], check=True, capture_output=True)
    native = (ROOT / 'native/shared/src/cuteafd_native.cc').read_text()
    assert 'getenv("CUTEAFD_RDMA_TRAFFIC_CLASS")' not in native
    assert 'attr.ah_attr.grh.traffic_class = cuteafd::kRdmaTrafficClass;' in native
    assert 'attr.ah_attr.sl = cuteafd::kRdmaTrafficClass >> 5;' in native
    launcher = (ROOT / 'scripts/launch/run-family.sh').read_text()
    assert 'RDMA_TRAFFIC_CLASS' not in launcher

def test_switch_files_keep_updated_baseline_and_separate_dscp_only():
    setup = (ROOT / 'scripts/fabric/mikrotik-crs8xx-roce-lossless.rsc').read_text()
    rollback = (ROOT / 'scripts/fabric/mikrotik-crs8xx-roce-rollback.rsc').read_text()
    for text in [setup, rollback]:
        assert 'ecn=yes' not in text and 'wred=yes' not in text
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line and not line.startswith('#'):
                assert index > 0 and lines[index - 1].startswith('#'), line
    assert setup.count('trust-l2=keep trust-l3=keep') == 7
    assert setup.count('egress-rate-queue3=100.0Gbps') == 6
    assert setup.count('egress-rate-queue3=400.0Gbps') == 1
    assert setup.count('automap=no') == 2
    assert 'map ip add dscp=26 profile=cuteafd-roce' in setup
    assert 'map ip add dscp=48 profile=cuteafd-cnp' in setup
    assert 'traffic-class=3 rx=yes tx=yes' in setup
    assert 'set 1 ecn=no schedule=low-priority-group weight=2 wred=no' in rollback
    assert rollback.count('!egress-rate-queue3') == 33
    assert rollback.count('pfc=disabled') == 32
    assert 'profile="cuteafd-roce"' in rollback


SCRIPT = ROOT / 'scripts/host/tune.sh'
ORIGINAL = {
    'netdev': 'roce0', 'rdma': ['mlx5_0', '1'], 'trust': 'pcp',
    'pfc': ['0'] * 8, 'buffers': ['1'] * 8,
    'sizes': ['0', '459360'] + ['0'] * 6,
    'app': {'dscp_prio': [[7, 2], [8, 1]], 'ethtype_prio': [[35078, 0]]},
    'pause': {'autoneg': 'off', 'rx': 'on', 'tx': 'on'},
}


def put(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def nic_state(root, state, netdev='roce0'):
    qos = ('DCBX mode: OS controlled\n'
           f'Priority trust state: {state["trust"]}\n'
           f'Receive buffer size (bytes): {",".join(state["sizes"])},max_buffer_size=2039616\n'
           'PFC configuration:\n priority 0 1 2 3 4 5 6 7\n'
           f' enabled {" ".join(state["pfc"])}\n buffer {" ".join(state["buffers"])}\n'
           'tc: 3 ratelimit: unlimited, tsa: vendor\n priority: 3\n'
           'tc: 6 ratelimit: unlimited, tsa: vendor\n priority: 6\n')
    put(root, f'class/net/{netdev}/mlnx_qos.txt', qos)
    put(root, f'class/net/{netdev}/app.json', json.dumps(state['app']))
    pause = state['pause']
    put(root, f'class/net/{netdev}/pause.txt',
        f'Autonegotiate: {pause["autoneg"]}\nRX: {pause["rx"]}\nTX: {pause["tx"]}\n')


def fabric_fixture(root, backup=True):
    base = 'class/infiniband/mlx5_0'
    put(root, base + '/ports/1/state', '4: ACTIVE\n')
    put(root, base + '/ports/1/link_layer', 'Ethernet\n')
    put(root, base + '/ports/1/gid_attrs/ndevs/0', 'roce0\n')
    (root / base / 'device/net/roce0').mkdir(parents=True)
    put(root, 'class/net/roce0/ip.json', json.dumps([
        {'addr_info': [{'local': '10.55.0.22'}, {'local': '10.55.1.22'}]}]))
    nic_state(root, ORIGINAL)
    if backup:
        return put(root, 'var/lib/cuteafd/fabric-lossless/roce0.json', json.dumps(ORIGINAL) + '\n')


def run_fabric(root, action='reapply', role='coordinator', dry_run=True):
    return subprocess.run(
        ['bash', str(SCRIPT), 'fabric-lossless', action, '--role', role] +
        (['--dry-run'] if dry_run else []),
        env=dict(os.environ, TUNE_SYSFS_ROOT=str(root)),
        text=True, capture_output=True, timeout=70)


def commands(result):
    return [line for line in result.stdout.splitlines() if line.startswith('DRY-RUN command ')]


def expected_commands():
    prefix = 'DRY-RUN command '
    mappings = ' '.join(f'--dscp2prio=set,{d},{3 if d == 26 else 6 if d == 48 else 0}' for d in range(64))
    return [prefix + text for text in [
        'ethtool -A roce0 autoneg off rx off tx off',
        'mlnx_qos -i roce0 --buffer_size 229376,229376,0,0,0,0,0,0',
        'mlnx_qos -i roce0 --prio2buffer 0,0,0,1,0,0,0,0',
        'dcb app flush dev roce0 dscp-prio',
        'mlnx_qos -i roce0 ' + mappings,
        'mlnx_qos -i roce0 --trust dscp',
        'mlnx_qos -i roce0 --pfc 0,0,0,1,0,0,0,0',
    ]]


def applied_state():
    return {**ORIGINAL, 'trust': 'dscp', 'pfc': ['0'] * 3 + ['1'] + ['0'] * 4,
            'buffers': ['0'] * 3 + ['1'] + ['0'] * 4,
            'sizes': ['229376', '229376'] + ['0'] * 6,
            'app': {'dscp_prio': [[d, 3 if d == 26 else 6 if d == 48 else 0] for d in reversed(range(64))]},
            'pause': {'autoneg': 'off', 'rx': 'off', 'tx': 'off'}}


def test_reapply_reset_nic_exact_policy_and_immutable_backup(tmp_path):
    backup = fabric_fixture(tmp_path)
    before, stat = backup.read_bytes(), backup.stat()
    result = run_fabric(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'state == saved original; would apply policy' in result.stdout
    assert commands(result) == expected_commands()
    assert backup.read_bytes() == before
    assert (backup.stat().st_ino, backup.stat().st_mtime_ns) == (stat.st_ino, stat.st_mtime_ns)
    assert list(backup.parent.iterdir()) == [backup]
    backup.unlink()
    apply = run_fabric(tmp_path, 'apply')
    assert apply.returncode == 0, apply.stderr
    assert commands(apply) == commands(result)


def test_reapply_applied_nic_noop(tmp_path):
    backup = fabric_fixture(tmp_path)
    before = backup.read_bytes(), backup.stat().st_mtime_ns
    nic_state(tmp_path, applied_state())
    result = run_fabric(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'roce0: already applied' in result.stdout
    assert not commands(result)
    assert (backup.read_bytes(), backup.stat().st_mtime_ns) == before


@pytest.mark.parametrize('change,field', [
    ({'trust': 'dscp'}, 'trust'),
    ({'pfc': ['1'] + ['0'] * 7}, 'pfc'),
    ({'buffers': ['0'] * 8}, 'buffers'),
    ({'sizes': ['459360'] + ['0'] * 7}, 'sizes'),
    ({'app': {'dscp_prio': [[7, 1]]}}, 'dscp_prio'),
    ({'pause': {'autoneg': 'on', 'rx': 'on', 'tx': 'on'}}, 'pause'),
])
def test_reapply_unknown_state_diff_and_immutable_backup(tmp_path, change, field):
    backup = fabric_fixture(tmp_path)
    before = backup.read_bytes(), backup.stat().st_mtime_ns
    nic_state(tmp_path, {**ORIGINAL, **change})
    result = run_fabric(tmp_path)
    assert result.returncode != 0
    assert 'roce0: unknown NIC state; refusing reapply' in result.stderr
    assert f'{field}: current=' in result.stderr
    assert 'saved original=' in result.stderr and 'policy=' in result.stderr
    assert not commands(result)
    assert (backup.read_bytes(), backup.stat().st_mtime_ns) == before


def test_reapply_no_backup_refuses_even_applied_nic(tmp_path):
    fabric_fixture(tmp_path, backup=False)
    nic_state(tmp_path, applied_state())
    result = run_fabric(tmp_path)
    assert result.returncode != 0
    assert 'no pending NIC backup' in result.stderr
    assert not commands(result)


def test_reapply_waits_for_fabric_address(tmp_path):
    fabric_fixture(tmp_path)
    address = tmp_path / 'class/net/roce0/ip.json'
    contents = address.read_text()
    address.write_text('[]')

    def assign_address():
        time.sleep(1.5)
        staged = address.with_suffix('.new')
        staged.write_text(contents)
        staged.replace(address)

    worker = threading.Thread(target=assign_address)
    worker.start()
    try:
        result = run_fabric(tmp_path)
    finally:
        worker.join()
    assert result.returncode == 0, result.stderr
    assert commands(result) == expected_commands()


def test_reapply_unreadable_backup_reports_permission(tmp_path):
    if os.geteuid() == 0:
        pytest.skip('root bypasses fixture directory permissions')
    backup = fabric_fixture(tmp_path)
    backup.parent.chmod(0o000)
    try:
        result = run_fabric(tmp_path)
    finally:
        backup.parent.chmod(0o700)
    assert result.returncode != 0
    assert 'permission denied' in result.stderr
    assert 'cannot verify saved original state, even in --dry-run' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not commands(result)


@pytest.mark.parametrize('role', ['coordinator', 'spark'])
def test_fabric_persist_prints_root_owned_unit_without_writes(tmp_path, role):
    result = run_fabric(tmp_path, 'persist', role, dry_run=False)
    assert result.returncode == 0, result.stderr
    assert f'ExecStart=/usr/bin/python3 -I /usr/local/lib/cuteafd/fabric-lossless.py reapply --role {role}' in result.stdout
    assert 'install -o root -g root -m 0755' in result.stdout
    assert 'cuteafd-fabric-lossless.service' in result.stdout
    assert 'After=network-online.target' in result.stdout
    assert 'Wants=network-online.target' in result.stdout
    assert 'ConditionFileIsExecutable=/usr/local/lib/cuteafd/fabric-lossless.py' in result.stdout
    assert 'Type=oneshot' in result.stdout and 'RemainAfterExit=yes' in result.stdout
    assert '/home/' not in result.stdout and str(tmp_path) not in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('action', ['apply', 'reapply'])
def test_fabric_mutation_refuses_sparknest_owned_nic(tmp_path, action):
    backup = fabric_fixture(tmp_path, backup=action == 'reapply')
    other = put(tmp_path, 'var/lib/sparknest/fabric-qos/roce0.json', '{"owner":"sparknest"}\n')
    before = other.read_bytes(), other.stat().st_mtime_ns
    result = run_fabric(tmp_path, action)
    assert result.returncode != 0
    assert 'roce0: refusing NIC changes; owner=SparkNest' in result.stderr
    assert str(other) in result.stderr
    assert not commands(result)
    assert (other.read_bytes(), other.stat().st_mtime_ns) == before
    if backup:
        assert json.loads(backup.read_text()) == ORIGINAL


def test_fabric_check_accepts_sparknest_superset_and_reports_owner(tmp_path):
    fabric_fixture(tmp_path)
    put(tmp_path, 'var/lib/sparknest/fabric-qos/roce0.json', '{}')
    state = applied_state()
    state['pfc'][2] = '1'
    state['buffers'] = ['0', '0', '1', '2', '0', '0', '0', '0']
    state['sizes'] = ['153088'] * 3 + ['0'] * 5
    state['app']['dscp_prio'] = [[d, 2 if d == 18 else p] for d, p in state['app']['dscp_prio']]
    nic_state(tmp_path, state)
    result = run_fabric(tmp_path, 'check', dry_run=False)
    assert result.returncode == 0, result.stderr
    assert 'roce0: cuteafd-compliant; owner=SparkNest' in result.stdout
    assert 'command ' not in result.stdout
    state['pfc'][3] = '0'
    nic_state(tmp_path, state)
    result = run_fabric(tmp_path, 'check', dry_run=False)
    assert result.returncode == 0, result.stderr
    assert 'roce0: NOT cuteafd-compliant; owner=SparkNest' in result.stdout


def test_fabric_check_reads_iproute2_dscp_triples(tmp_path):
    # Real `dcb -j app show` output carries a third flag element per mapping.
    fabric_fixture(tmp_path)
    put(tmp_path, 'var/lib/sparknest/fabric-qos/roce0.json', '{}')
    state = applied_state()
    state['pfc'][2] = '1'
    state['buffers'] = ['0', '0', '1', '2', '0', '0', '0', '0']
    state['sizes'] = ['153088'] * 3 + ['0'] * 5
    state['app']['dscp_prio'] = [[d, 2 if d == 18 else p, ' '] for d, p in state['app']['dscp_prio']]
    nic_state(tmp_path, state)
    result = run_fabric(tmp_path, 'check', dry_run=False)
    assert result.returncode == 0, result.stderr
    assert 'roce0: cuteafd-compliant; owner=SparkNest' in result.stdout
    state['app']['dscp_prio'] = [[d, p, 7] for d, p, _ in state['app']['dscp_prio']]
    nic_state(tmp_path, state)
    result = run_fabric(tmp_path, 'check', dry_run=False)
    assert result.returncode != 0 and 'invalid DSCP mappings' in result.stderr
