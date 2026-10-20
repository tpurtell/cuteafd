"""Build seeds off the RoCE fabric (kakapo): seed choice, lock order, transfer selection."""
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cards = _load('wip_cards_seed', ROOT / 'scripts/bench/wip-cards.py')
hwlock = _load('hwlock_seed', ROOT / 'scripts/hw/hwlock.py')
transfer = _load('seed_transfer', ROOT / 'scripts/build/seed-transfer.py')
WIP = (ROOT / 'wip.sh').read_text()


@pytest.fixture
def entry(tmp_path):
    base = tmp_path / 'model.config'
    base.write_text('MODEL_ID=test/model\n')
    return dict(name='mimo26-flash-max', family='mimo_v2', config=str(base), gpus=[0, 1],
                sparks=['ostrich', 'dodo', 'emu', 'kiwi'], set={'SPARK_COUNT': '4'})


def test_kakapo_is_a_seed_choice_but_never_a_serving_host():
    assert 'kakapo' in cards.SEEDS and 'kakapo' not in cards.HOSTS
    assert {'rhea', 'moa'} <= set(cards.SEEDS)


def _staged(cfg):
    return [cfg[f'SPARK_{i}_HOST'] for i in range(int(cfg['SPARK_COUNT']))]


def _build(arms, cards_, tmp_path, monkeypatch, seed, separate=True):
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    monkeypatch.setattr(cards, 'build_slot_capable', lambda rev: True)
    monkeypatch.setattr(cards, 'seed_option_capable', lambda rev: separate)
    monkeypatch.setattr(cards, 'toolchain_hash', lambda rev: '0123456789abcdef')
    cards.build_arms({'base': 'HEAD'}, arms, cards_, tmp_path, 'task', True, seed)
    return cards.config(tmp_path / 'build-base.config')


def test_kakapo_seed_is_never_a_staged_spark(entry, tmp_path, monkeypatch, capsys):
    # #117: the four-Spark gate card with the build-only seed stages its own four
    # Sparks; the seed goes to wip.sh as WIP_SEED_HOST, never as a SPARK_* rank.
    cfg = _build(cards.arms_from(['base=b:s']), [entry], tmp_path, monkeypatch, 'kakapo')
    assert _staged(cfg) == ['ostrich', 'dodo', 'emu', 'kiwi'] and cfg['SPARK_COUNT'] == '4'
    assert 'kakapo' not in cfg.values() and '0.0.0.0' not in cfg.values()
    assert cfg['SPARK_0_LANE_A'] == '10.55.0.1'
    assert '# build seed base: kakapo (WIP_SEED_HOST); Spark targets (4): ostrich,dodo,emu,kiwi' in capsys.readouterr().out


def test_six_spark_card_with_kakapo_stages_six_ranks(entry, tmp_path, monkeypatch):
    six = {**entry, 'name': 'mimo26-pro-max', 'sparks': list(cards.HOSTS), 'set': {'SPARK_COUNT': '6'}}
    cfg = _build(cards.arms_from(['base=b:s']), [six], tmp_path, monkeypatch, 'kakapo')
    assert cfg['SPARK_COUNT'] == '6' and _staged(cfg) == list(cards.HOSTS)
    assert not any(k.startswith('SPARK_6_') for k in cfg)


def test_seed_inside_the_union_is_one_host(entry, tmp_path, monkeypatch):
    six = {**entry, 'sparks': list(cards.HOSTS), 'set': {'SPARK_COUNT': '6'}}
    cfg = _build(cards.arms_from(['base=b:s']), [six], tmp_path, monkeypatch, 'moa')
    assert _staged(cfg).count('moa') == 1 and cfg['SPARK_COUNT'] == '6'


def test_pool_seed_outside_the_union_fills_the_rank_count_first(entry, tmp_path, monkeypatch):
    # A five-host union pads to six: a pool-Spark seed is the preferred filler, so it
    # is one host that both builds and receives the slot.
    five = {**entry, 'sparks': ['ostrich', 'dodo', 'emu', 'kiwi', 'moa'], 'set': {'SPARK_COUNT': '5'}}
    cfg = _build(cards.arms_from(['base=b:s']), [five], tmp_path, monkeypatch, 'rhea')
    assert sorted(_staged(cfg)) == sorted(cards.HOSTS)
    assert cards.staging_hosts({'ostrich', 'dodo', 'emu', 'kiwi', 'moa'}, 'kakapo', True) == [
        'ostrich', 'dodo', 'emu', 'kiwi', 'moa', 'rhea']


def test_legacy_wip_sh_keeps_the_seed_as_rank_zero(entry, tmp_path, monkeypatch):
    # A revision whose wip.sh predates WIP_SEED_HOST builds on SPARK_0_HOST.
    monkeypatch.setattr(cards, 'seed_lanes', lambda host, dry: ('172.22.2.7',) * 2 if host == 'kakapo'
                        else (f'10.55.0.{cards.HOSTS.index(host) + 1}', f'10.55.1.{cards.HOSTS.index(host) + 1}'))
    cfg = _build(cards.arms_from(['base=b:s']), [entry], tmp_path, monkeypatch, 'kakapo', separate=False)
    assert _staged(cfg)[0] == 'kakapo' and set(entry['sparks']) <= set(_staged(cfg)[1:])
    assert cfg['SPARK_0_LANE_A'] == '172.22.2.7'


def test_seed_option_capability_detection(monkeypatch):
    shown = iter(['seed_option="${WIP_SEED_HOST:-}"', 'seed_host="${wip_hosts[0]}"'])
    monkeypatch.setattr(cards.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=next(shown)))
    assert cards.seed_option_capable('new') and not cards.seed_option_capable('old')
    assert 'WIP_SEED_HOST' in WIP


def test_seed_lanes_resolve_pool_rails_without_ssh_and_dry_runs_never_probe(monkeypatch):
    monkeypatch.setattr(cards, 'run', lambda cmd: pytest.fail('no probe expected'))
    assert cards.seed_lanes('moa', False) == ('10.55.0.6', '10.55.1.6')
    assert cards.seed_lanes('kakapo', True) == ('0.0.0.0', '0.0.0.0')


def test_default_seed_is_the_build_only_kakapo():
    assert cards.DEFAULT_SEED == 'kakapo'


def test_cleanup_covers_seed_hosts_and_the_instance_nest_area(entry, monkeypatch, capsys):
    arms = cards.arms_from(['base=inst:s'])
    cards.cleanup(arms, 'task', True)
    out = capsys.readouterr().out
    assert 'ssh -o BatchMode=yes kakapo' in out
    assert 'nest rm -r' in out and '/mnt/sparknest/build-cache/wip/inst' in out
    # The area name matches what wip.sh writes for this instance.
    assert "printf '/mnt/sparknest/build-cache/wip/%s/%s\\n' \"${WIP_INSTANCE:-_default}\" \"$slot\"" in WIP


def test_seed_lock_is_ordered_after_every_serving_lock():
    assert hwlock.canonical(['kakapo', 'sparks', 'moa']) == ['sparks', 'moa', 'kakapo']


IP_RAPTOR = '''3: enp172s0    inet 172.22.2.12/16 metric 100 brd 172.22.255.255 scope global dynamic enp172s0
5: enp1s0np0    inet 10.55.0.22/24 metric 100 brd 10.55.0.255 scope global dynamic enp1s0np0
6: tailscale0    inet 100.97.225.1/32 scope global tailscale0
7: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
'''
IP_KAKAPO = '''2: enP7s7    inet 172.22.2.7/16 metric 100 brd 172.22.255.255 scope global dynamic enP7s7
8: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
9: tailscale0    inet 100.97.190.23/32 scope global tailscale0
'''


def test_lan_addr_prefers_the_shared_physical_subnet_over_tailscale_and_docker():
    assert transfer.lan_addr(IP_RAPTOR, IP_KAKAPO) == '172.22.2.7'
    assert transfer.lan_addr(IP_RAPTOR, IP_KAKAPO.replace('172.22.2.7/16', '192.168.9.7/24')) is None


def _bash(script, tmp_path, tools):
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    for name, body in tools.items():
        (bindir / name).write_text('#!/bin/sh\n' + body + '\n')
        (bindir / name).chmod(0o755)
    trace = tmp_path / 'trace'
    trace.touch()
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + script], capture_output=True, text=True, timeout=60,
                            env={**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'TRACE': str(trace)})
    return result, trace.read_text()


def _function(name):
    start = WIP.index(name + '() {')
    return WIP[start:WIP.index('\n}\n', start) + 3]


@pytest.mark.parametrize('rdma,setting,mounted,expected', [
    (1, 'auto', True, 'stream'), (0, 'auto', True, 'nest'), (0, 'auto', False, 'relay'),
    (0, 'stream', True, 'relay'), (1, 'stream', True, 'stream'), (1, 'nest', True, 'nest')])
def test_slot_transfer_mode_selection(tmp_path, rdma, setting, mounted, expected):
    script = _function('wip_slot_transfer_mode') + f'''
seed_host=kakapo wip_target_hosts=(ostrich dodo) wip_seed_rdma={rdma} wip_seed_transfer={setting}
wip_slot_transfer_mode
'''
    result, _ = _bash(script, tmp_path, {'ssh': 'exit ' + ('0' if mounted else '1')})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_seed_leg_of_a_stream_is_compressed_over_the_lan_address(tmp_path):
    script = WIP[WIP.index('stream_between_hosts() {'):WIP.index('distribute_spark_dev_image() {')] + '''
release_ssh() { printf 'ssh %s\\n' "$*" >> "$TRACE"; }
release_bulk_ssh_opts=(-c aes128-gcm@openssh.com -o Compression=no)
''' + _release_bulk_ssh() + '''
seed_host=kakapo wip_seed_rdma=0 wip_seed_addr=172.22.2.7
stream_between_hosts kakapo 'tar -cf - .' ostrich 'tar -xf -'
'''
    result, trace = _bash(script, tmp_path, {'ssh': 'printf "ssh %s\\n" "$*" >> "$TRACE"'})
    assert result.returncode == 0, result.stderr
    # The two ends of the pipe trace concurrently, so their lines arrive in either order.
    first, second = sorted(trace.splitlines(), key=lambda line: 'ostrich' in line.split(' zstd')[0])
    assert '-c aes128-gcm@openssh.com -o Compression=no' in first
    assert '-o HostName=172.22.2.7 -o HostKeyAlias=kakapo kakapo' in first and first.endswith('tar -cf - . | zstd -T8 -3 -q')
    assert second.startswith('ssh -o BatchMode=yes ostrich zstd -d -q | tar -xf -')


def test_rdma_seed_keeps_rdmapipe_and_plain_pipes(tmp_path):
    script = WIP[WIP.index('stream_between_hosts() {'):WIP.index('distribute_spark_dev_image() {')] + '''
seed_host=moa wip_seed_rdma=1 wip_seed_addr=
stream_between_hosts moa 'a' ostrich 'b'
'''
    result, trace = _bash(script, tmp_path, {'ssh': 'printf "ssh %s\\n" "$*" >> "$TRACE"'})
    assert result.returncode == 0, result.stderr
    assert 'rdmapipe --send' in trace and 'zstd' not in trace and 'aes128-gcm' not in trace


def _release_bulk_ssh():
    common = (ROOT / 'scripts/lib/release-common.sh').read_text()
    start = common.index('release_bulk_ssh() {')
    return common[start:common.index('\n}\n', start) + 3]


def test_source_sync_to_a_non_rdma_seed_uses_rsync_over_the_fast_cipher_and_lan():
    sync = WIP[WIP.index('sync_seed_source() {'):WIP.index('# Native/AOT outputs are cached by input key')]
    assert '[[ "$wip_seed_rdma" == 1 ]] && command -v rdmasync' in sync
    assert ('rsync -a --delete -e "$release_rsh ${release_bulk_ssh_opts[*]} -o HostName=$wip_seed_addr '
            '-o HostKeyAlias=$seed_host"') in sync


def test_nest_distribution_writes_once_replicates_first_hop_then_fans_out():
    nest = WIP[WIP.index('distribute_expert_slot_nest() {'):WIP.index('\ndistribute_expert_slot() {')]
    write, first, fanout, extract = (nest.index(s) for s in (
        'zstd -T0 -1 -q', 'nest replicate --hosts \'$1\'', 'nest replicate --hosts \'$(IFS=,', 'zstd -d -q <"$archive"'))
    assert write < first < fanout < extract
    assert 'nest seal "$archive"' in nest and 'nest_settle "$archive"' in nest
    # The relay path crosses the seed's link once, then fans out from a RoCE host.
    relay = WIP[WIP.index('    relay)'):WIP.index('    stream) distribute_expert_slot_from')]
    assert 'distribute_expert_slot_from "$seed_host" "${targets[0]}"' in relay
    assert 'distribute_expert_slot_from "${targets[0]}" "${targets[@]:1}"' in relay


def test_native_nest_tier_is_bounded_and_wraps_the_host_store():
    expert = WIP[WIP.index('build_expert() ('):WIP.index('\n# Role builders end here.')]
    order = [expert.index(s) for s in (
        'native_store_nest "$seed_host" aarch64 "$seed_root" expert import',
        'native_store_sync "$seed_host" "$seed_root" aarch64 expert import',
        'native_store_sync "$seed_host" "$seed_root" aarch64 expert publish',
        'native_store_nest "$seed_host" aarch64 "$seed_root" expert publish')]
    assert order == sorted(order)
    tier = WIP[WIP.index('native_store_nest() {'):WIP.index('# native_store_has HOST ROOT OUTPUT')]
    assert 'CUTEAFD_WIP_NEST_NATIVE_KEEP:-8' in tier and 'tail -n +"$((keep + 1))"' in tier
    assert '/mnt/sparknest/build-cache/wip-native/$arch/$scope' in tier


def test_dev_images_reach_build_seeds_compressed_when_an_end_lacks_rdma(tmp_path):
    text = (ROOT / 'scripts/build/build-dev-images.sh').read_text()
    assert 'seed_hosts_csv=kakapo' in text
    copy = text[text.index('copy_spark_image() {'):text.index('\n}\n', text.index('copy_spark_image() {')) + 3]
    script = '''
release_ssh() { printf 'ssh %s\\n' "$*" >> "$TRACE"; }
release_host_has_rdma() { [[ "$1" != kakapo ]]; }
release_bulk_ssh_opts=(-c aes128-gcm@openssh.com -o Compression=no)
''' + _release_bulk_ssh() + f'''
repo_root={str(ROOT)!r} spark_tag=img
python3() {{ printf '172.22.2.7\\n'; }}
''' + copy + '''
copy_spark_image ostrich kakapo
echo ---- >> "$TRACE"
copy_spark_image ostrich dodo
'''
    result, trace = _bash(script, tmp_path, {'ssh': 'printf "ssh %s\\n" "$*" >> "$TRACE"'})
    assert result.returncode == 0, result.stderr
    compressed, plain = trace.split('----\n')
    assert "docker image save 'img' | zstd -T8 -3 -q" in compressed
    assert '-o HostName=172.22.2.7 -o HostKeyAlias=kakapo kakapo zstd -d -q | docker image load' in compressed
    assert 'zstd' not in plain and 'aes128-gcm' not in plain


def test_seed_env_passes_the_seed_and_its_lan_address_only_to_capable_spark_builds(monkeypatch):
    monkeypatch.setattr(cards, 'run', lambda cmd: SimpleNamespace(stdout='172.22.2.7\n'))
    both = dict(seed_separate=True, build_role='both')
    assert cards.seed_env(both, 'kakapo', False) == {'WIP_SEED_HOST': 'kakapo', 'WIP_SEED_ADDR': '172.22.2.7'}
    assert cards.seed_env(both, 'moa', False) == {'WIP_SEED_HOST': 'moa'}
    assert cards.seed_env({**both, 'seed_separate': False}, 'kakapo', False) == {}
    assert cards.seed_env({**both, 'build_role': 'coordinator'}, 'kakapo', False) == {}


SIX = '\n'.join([f'SPARK_{i}_HOST={h}\nSPARK_{i}_LANE_A=10.55.0.{i + 1}\nSPARK_{i}_LANE_B=10.55.1.{i + 1}'
                 for i, h in enumerate(cards.HOSTS)])


def _wip_dry_run(tmp_path, spark_lines, count, extra_args=(), env=None):
    config = tmp_path / 'build.config'
    config.write_text(f'FAMILY=mimo_v2\nMODEL_ID=XiaomiMiMo/MiMo-V2.6-Pro\nSPARK_COUNT={count}\nCOORDINATOR_GPU=1\nRTX_GPUS=1\n{spark_lines}\n')
    stub = tmp_path / 'bin'
    stub.mkdir(exist_ok=True)
    marker = tmp_path / 'called'
    for name in ('docker', 'ssh', 'rsync', 'nvidia-smi'):
        (stub / name).write_text(f'#!/usr/bin/env bash\necho "{name} $*" >>"{marker}"\nexit 1\n')
        (stub / name).chmod(0o755)
    run_env = {k: v for k, v in os.environ.items() if k not in ('WIP_SEED_HOST', 'WIP_SEED_ADDR')}
    run_env.update(PATH=f'{stub}:{os.environ["PATH"]}', **(env or {}))
    result = subprocess.run([str(ROOT / 'wip.sh'), '--config', str(config), '--dry-run', *extra_args],
                            cwd=ROOT, capture_output=True, text=True, timeout=120, env=run_env)
    assert result.returncode == 0, result.stderr
    assert not marker.exists(), marker.read_text()
    return result.stdout


def test_wip_dry_run_six_targets_with_a_build_only_seed(tmp_path):
    out = _wip_dry_run(tmp_path, SIX, 6, env={'WIP_SEED_HOST': 'kakapo', 'WIP_SEED_ADDR': '172.22.2.7'})
    assert 'Spark hosts (6): ostrich,dodo,emu,kiwi,rhea,moa' in out
    assert 'seed host: kakapo (build-only, transfer auto via 172.22.2.7)' in out
    assert 'Spark targets (6): ostrich,dodo,emu,kiwi,rhea,moa' in out
    # --seed is the same parameter as WIP_SEED_HOST.
    assert 'seed host: kakapo (build-only' in _wip_dry_run(tmp_path, SIX, 6, ['--seed', 'kakapo'])


def test_wip_dry_run_seed_inside_the_union_is_one_host(tmp_path):
    out = _wip_dry_run(tmp_path, SIX, 6, ['--seed', 'moa'])
    assert 'seed host: moa (serving' in out
    assert 'Spark targets (5): ostrich,dodo,emu,kiwi,rhea' in out


def test_wip_dry_run_default_seed_is_build_only_kakapo(tmp_path):
    four = '\n'.join(SIX.splitlines()[:12])
    out = _wip_dry_run(tmp_path, four, 4)
    assert 'Spark hosts (4): ostrich,dodo,emu,kiwi' in out
    assert 'seed host: kakapo (build-only' in out
    assert 'Spark targets (4): ostrich,dodo,emu,kiwi' in out


def test_build_only_seed_gets_a_build_container_and_never_a_distribution():
    hosts = WIP[WIP.index('# The seed builds the Spark slot; the configured Sparks'):WIP.index('# Opt-in replicated-group Spark expert roles')]
    script = 'set -euo pipefail\nwip_hosts=(ostrich dodo)\nseed_option=kakapo\n' + hosts + '''
printf '%s|%s|%s\\n' "${wip_target_hosts[*]}" "${wip_container_hosts[*]}" "$wip_seed_serves"
seed_option=dodo
''' + hosts + '''
printf '%s|%s|%s\\n' "${wip_target_hosts[*]}" "${wip_container_hosts[*]}" "$wip_seed_serves"
'''
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['ostrich dodo|kakapo ostrich dodo|0', 'ostrich|ostrich dodo|1']
    # Distribution and fingerprint checks iterate the targets; containers, process
    # checks and --recreate cover the seed's build container too.
    assert WIP.count('for host in "${wip_target_hosts[@]}"; do') == 3
    assert WIP.count('"${wip_container_hosts[@]}"') == 6


def test_legacy_wip_sh_with_a_six_spark_union_and_an_outside_seed_is_refused_early(entry, tmp_path, monkeypatch):
    six = {**entry, 'sparks': list(cards.HOSTS), 'set': {'SPARK_COUNT': '6'}}
    with pytest.raises(ValueError, match='predates WIP_SEED_HOST'):
        _build(cards.arms_from(['base=b:s']), [six], tmp_path, monkeypatch, 'kakapo', separate=False)
    # A pool seed inside the union is still one of the six.
    assert int(_build(cards.arms_from(['base=b:s']), [six], tmp_path, monkeypatch, 'moa', separate=False)['SPARK_COUNT']) == 6
