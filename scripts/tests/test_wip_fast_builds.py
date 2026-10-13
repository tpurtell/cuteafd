"""Content-keyed WIP reuse: sealed slot adoption, native-output cache, fast source checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
REUSE = ROOT / 'scripts/build/wip-slot-reuse.py'
KEY = 'a' * 64


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sealed_slot(home: Path, instance: str, slot: str, role: str = 'coordinator', key: str = KEY) -> Path:
    """A slot laid out as finalize-wip-slot.sh seals it."""
    path = home / '.cache/cuteafd/builds' / f'wip-{instance}' / 'slots' / slot / role
    artifacts = path / 'workspace/.cuteafd-wip'
    artifacts.mkdir(parents=True)
    (artifacts / 'cuteafd').write_bytes(b'daemon')
    (artifacts / 'libcuteafd_native.so').write_bytes(b'native')
    (artifacts / 'ARTIFACT_SHA256SUMS').write_text(
        f'{sha(b"daemon")}  cuteafd\n{sha(b"native")}  libcuteafd_native.so\n')
    (path / 'workspace/README.md').write_text('source\n')
    (path / 'SOURCE_SHA256SUMS').write_text(f'{sha(b"source" + bytes([10]))}  ./README.md\n')
    meta = {'schema': 1, 'wip_instance': instance, 'wip_root': '', 'slot': slot, 'role': role,
            'build_key': key,
            'source_manifest_sha256': sha((path / 'SOURCE_SHA256SUMS').read_bytes()),
            'artifact_manifest_sha256': sha((artifacts / 'ARTIFACT_SHA256SUMS').read_bytes())}
    data = (json.dumps(meta, indent=2, sort_keys=True) + '\n').encode()
    (path / 'META.json').write_bytes(data)
    (path / 'FINGERPRINT').write_text(sha(data) + '\n')
    return path


def reuse(home, *args):
    return subprocess.run(['python3', str(REUSE), *args], env={**os.environ, 'HOME': str(home)},
                          capture_output=True, text=True)


def seal_ok(slot: Path):
    meta = (slot / 'META.json').read_bytes()
    assert sha(meta) == (slot / 'FINGERPRINT').read_text().strip()
    return json.loads(meta)


def test_adopts_a_verified_slot_and_reseals_only_its_identity(tmp_path):
    source = sealed_slot(tmp_path, 'base', 'gate')
    root = tmp_path / '.cache/cuteafd/builds/wip-mine'
    found = reuse(tmp_path, 'find', '--key', KEY, '--role', 'coordinator', '--slot', 'gate', '--exclude-root', str(root))
    assert found.returncode == 0, found.stderr
    assert Path(found.stdout.strip()) == source
    adopted = reuse(tmp_path, 'adopt', '--key', KEY, '--role', 'coordinator', '--source', str(source),
                    '--root', str(root), '--slot', 'gate', '--instance', 'mine', '--wip-root', str(root))
    assert adopted.returncode == 0, adopted.stderr
    target = root / 'slots/gate/coordinator'
    meta = seal_ok(target)
    assert (meta['wip_instance'], meta['slot'], meta['wip_root']) == ('mine', 'gate', str(root))
    assert meta['reused_from']['wip_instance'] == 'base'
    original = seal_ok(source)
    assert meta['artifact_manifest_sha256'] == original['artifact_manifest_sha256']
    assert meta['source_manifest_sha256'] == original['source_manifest_sha256']
    # Artifacts are shared by hardlink; the source slot's seal is untouched.
    assert os.stat(target / 'workspace/.cuteafd-wip/cuteafd').st_ino == os.stat(source / 'workspace/.cuteafd-wip/cuteafd').st_ino
    assert original['wip_instance'] == 'base'


@pytest.mark.parametrize('damage', ['artifact', 'source', 'meta', 'key'])
def test_refuses_any_slot_whose_seal_or_key_does_not_verify(tmp_path, damage):
    source = sealed_slot(tmp_path, 'base', 'gate', key=('b' * 64 if damage == 'key' else KEY))
    if damage == 'artifact':
        (source / 'workspace/.cuteafd-wip/libcuteafd_native.so').write_bytes(b'changed')
    elif damage == 'source':
        (source / 'workspace/README.md').write_text('edited\n')
    elif damage == 'meta':
        meta = json.loads((source / 'META.json').read_text())
        meta['build_key'] = KEY
        meta['slot'] = 'other'
        (source / 'META.json').write_text(json.dumps(meta))
    found = reuse(tmp_path, 'find', '--key', KEY, '--role', 'coordinator', '--slot', 'gate', '--exclude-root', str(tmp_path / 'x'))
    assert found.returncode == 1 and not found.stdout.strip()


def test_own_slot_with_the_key_is_reused_in_place(tmp_path):
    own = sealed_slot(tmp_path, 'mine', 'gate')
    sealed_slot(tmp_path, 'other', 'gate')
    root = tmp_path / '.cache/cuteafd/builds/wip-mine'
    found = reuse(tmp_path, 'find', '--key', KEY, '--role', 'coordinator', '--slot', 'gate', '--exclude-root', str(root))
    assert Path(found.stdout.strip()) == own
    before = (own / 'FINGERPRINT').read_text()
    adopted = reuse(tmp_path, 'adopt', '--key', KEY, '--role', 'coordinator', '--source', str(own),
                    '--root', str(root), '--slot', 'gate', '--instance', 'mine')
    assert adopted.returncode == 0 and 'already holds' in adopted.stdout
    assert (own / 'FINGERPRINT').read_text() == before


def test_roles_never_cross(tmp_path):
    sealed_slot(tmp_path, 'base', 'gate', role='spark-expert')
    found = reuse(tmp_path, 'find', '--key', KEY, '--role', 'coordinator', '--slot', 'gate', '--exclude-root', str(tmp_path / 'x'))
    assert found.returncode == 1


def test_wip_reuses_only_clean_committed_trees_and_records_the_key():
    wip = (ROOT / 'wip.sh').read_text()
    assert 'if [[ "$source_revision" != *dirty* && "$source_revision" != unknown && -z "$from_slot" ]]; then' in wip
    # The key covers the commit and tree, pinned submodules, config, scopes and each role's dev image.
    for part in ("printf 'schema 1\\ncommit %s\\ntree %s\\n'", "ls-tree -r HEAD | awk '$2 == \"commit\"",
                 "printf 'config %s\\n'", 'COORDINATOR_EXPERT_FAMILIES EXPERT_FAMILIES GENERIC_SPARK_COUNTS',
                 "'%s coordinator %s\\n'", "'%s spark-expert %s\\n'"):
        assert part in wip, part
    finalize = (ROOT / 'scripts/build/finalize-wip-slot.sh').read_text()
    assert 'metadata["build_key"] = build_key' in finalize and 'metadata["build_phases"] = phases' in finalize


def test_finalize_records_key_and_phases(tmp_path):
    script = (ROOT / 'scripts/build/finalize-wip-slot.sh').read_text()
    block = script[script.index("python3 - \"$incoming/META.json\""):script.index('sha256sum "$incoming/META.json"')]
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'BUILD_PHASES.tsv').write_text('coordinator\trust\t12\tkache enabled\ncoordinator\tnative\t3\treused\n')
    incoming = tmp_path / 'incoming'
    (incoming / 'workspace/.cuteafd-wip').mkdir(parents=True)
    (incoming / 'workspace/.cuteafd-wip/V41_EXPERT_TP_AOT.json').write_text('{"schema":1,"spark_tp_roles":[]}')
    values = dict(wip_instance='i', wip_root='', wip_build_key=KEY, slot='s', role='coordinator', image='img',
                  image_id='id', source_manifest_sha256='s', artifact_sha256='a', sparkinfer_revision='r',
                  transformers_revision='t', transformers_source_digest='d', build_output=str(output),
                  incoming=str(incoming))
    prefix = ''.join(f'{k}={v!r}\n' for k, v in values.items())
    subprocess.run(['bash', '-c', 'set -euo pipefail\n' + prefix + block], check=True)
    meta = json.loads((incoming / 'META.json').read_text())
    assert meta['build_key'] == KEY
    assert [(p['phase'], p['seconds']) for p in meta['build_phases']] == [('rust', 12), ('native', 3)]


def _wip_coordinator(tmp_path, hit):
    text = (ROOT / 'wip.sh').read_text()
    helpers = text[text.index('wip_hwlock() {'):text.index('\n}\n', text.index('wip_hwlock() {')) + 3]
    helpers += text[text.index('# Native/AOT outputs are cached by input key'):text.index('build_expert() (')]
    tools = tmp_path / 'tools'
    tools.mkdir()
    trace = tmp_path / 'trace'
    for name, body in {'docker': 'printf "docker %s\\n" "$*" >> "$TRACE"\nprintf "image-id\\n"',
                       'python3': 'printf "turn %s\\n" "$*" >> "$TRACE"'}.items():
        (tools / name).write_text('#!/bin/sh\n' + body + '\n')
        (tools / name).chmod(0o755)
    state = tmp_path / 'state'
    (state / 'output/coordinator').mkdir(parents=True)
    (state / 'output/coordinator/.native-cache-key').write_text('coordinator-120 ' + KEY + '\n')
    if hit:
        entry = state / 'cache/native-store/coordinator-120' / KEY
        entry.mkdir(parents=True)
        (entry / 'COMPLETE').write_text(KEY)
    setup = f'''set -euo pipefail
repo_root={str(ROOT)!r}
state_dir={str(state)!r}
sync_local_source() {{ :; }}
release_die() {{ exit 2; }}
cuteafd_compiler_cache_warn() {{ :; }}
slot=test coordinator_container=owned COORDINATOR_DOCKER_DEV=image bf16_families= audio_aot=OFF
CUTEAFD_WIP_EXPORT_LOCKS=on wip_export_locks=on wip_export_gpu_lock=gpu1
# The fair turn only (off-queue modes: test_wip_coordinator_export_modes.py).
CUTEAFD_WIP_COORDINATOR_EXPORT=turn
'''
    result = subprocess.run(['bash', '-c', setup + helpers + '\nbuild_coordinator'], capture_output=True, text=True,
                            env={**os.environ, 'HOME': str(tmp_path / 'home'), 'PATH': f'{tools}:{os.environ["PATH"]}',
                                 'TRACE': str(trace)})
    assert result.returncode == 0, result.stderr
    return result.stdout, trace.read_text().splitlines()


def test_native_cache_hit_takes_no_gpu_turn(tmp_path):
    stdout, lines = _wip_coordinator(tmp_path, hit=True)
    assert 'native cache hit, no GPU turn' in stdout
    assert not any(line.startswith('turn ') for line in lines)
    native = [line for line in lines if line.endswith('/wip/output/coordinator native')]
    assert len(native) == 1 and 'CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1' in native[0]


def test_native_cache_miss_queues_for_the_gpu(tmp_path):
    stdout, lines = _wip_coordinator(tmp_path, hit=False)
    assert 'queueing for gpu1' in stdout
    assert any(line.startswith('turn ') and 'hwlock.py --locks gpu1' in line for line in lines)


def test_native_store_moves_entries_between_instance_and_host_store(tmp_path):
    text = (ROOT / 'wip.sh').read_text()
    helpers = text[text.index('# Native/AOT outputs are cached by input key'):text.index('# native_store_has HOST ROOT OUTPUT')]
    home = tmp_path / 'home'
    first, second = tmp_path / 'first', tmp_path / 'second'
    for root in (first, second):
        (root / 'output/coordinator').mkdir(parents=True)
        (root / 'output/coordinator/.native-cache-key').write_text('coordinator-120 ' + KEY + '\n')
    entry = first / 'cache/native-store/coordinator-120' / KEY
    (entry / 'output').mkdir(parents=True)
    (entry / 'output/libcuteafd_native.so').write_bytes(b'native')
    (entry / 'COMPLETE').write_text(KEY)
    command = helpers + f'''
native_store_sync "" {str(first)!r} x86_64 coordinator publish
native_store_sync "" {str(second)!r} x86_64 coordinator import
'''
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + command], capture_output=True, text=True,
                            env={**os.environ, 'HOME': str(home)})
    assert result.returncode == 0, result.stderr
    assert 'published' in result.stdout and 'imported' in result.stdout
    imported = second / 'cache/native-store/coordinator-120' / KEY / 'output/libcuteafd_native.so'
    assert imported.read_bytes() == b'native'
    assert os.stat(imported).st_ino == os.stat(entry / 'output/libcuteafd_native.so').st_ino


def test_native_key_covers_every_native_input():
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    block = script[script.index('wip_native_key() {'):script.index('\n}\n', script.index('wip_native_key() {'))]
    for part in ('CUTEAFD_WIP_DEV_IMAGE_ID', "grep -E '^CUTEAFD_WIP_[A-Z0-9_]+='", '"$source_dir/third_party/"*.lock.json',
                 'find native python scripts/build', '"$role" "$cuda_arch"'):
        assert part in block, part
    # A hit restores every native output and nothing else; the daemon always comes from Cargo.
    assert 'native_items=(libcuteafd_native.so PROGRAMS.json V41_EXPERT_AOT.json V41_EXPERT_TP_AOT.json V41_FP8_AOT.json exl3 fp8)' in script
    assert script.index('cp -a "$native_entry/output/." "$output_dir/"') < script.index('install -m 0755 "$CARGO_TARGET_DIR/release/cuteafd"')


def test_source_checkout_initializes_only_build_submodules_from_local_stores():
    text = (ROOT / 'scripts/build/wip-source-checkout.sh').read_text()
    for name in ('third_party/sparkinfer', 'third_party/transformers', 'third_party/xgrammar'):
        assert f'submodule_from_local "$dest" {name}' in text
    assert 'deepseek-harness "' not in text and 'gptqmodel "' not in text
    assert 'remote set-url origin "$url"' in text
    assert 'verify-$name-source.py' in text


def test_capable_builds_hold_no_outer_build_lock_or_seed_gate(tmp_path, monkeypatch, capsys):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('wip_cards_fast', ROOT / 'scripts/bench/wip-cards.py')
    cards = importlib.util.module_from_spec(spec)
    sys.modules['wip_cards_fast'] = cards
    spec.loader.exec_module(cards)
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    monkeypatch.setattr(cards, 'toolchain_hash', lambda rev: '0123456789abcdef')
    monkeypatch.setattr(cards, 'build_slot_capable', lambda rev: True)
    config = tmp_path / 'card.config'
    config.write_text('SPARK_COUNT=4\n')
    entry = {'name': 'v41-max', 'family': 'deepseek_v41', 'config': str(config), 'sparks': ['ostrich', 'dodo', 'emu', 'kiwi']}
    arms = cards.arms_from(['cand=c:s'])
    cards.build_arms(['cand=new'], arms, [entry], tmp_path, 'task', True)
    text = capsys.readouterr().out
    assert 'build.lock' not in text and 'wip.sh --slot s --role both' in text
    assert arms['cand']['build_slot'] is True
    source = (ROOT / 'scripts/bench/wip-cards.py').read_text()
    assert "if build_role == 'both' and arm['build_lock_mode'] != 'native-phase':" in source
    wip = (ROOT / 'wip.sh').read_text()
    assert '"$repo_root/scripts/build/build-slot.sh" "${coordinator_build[@]}" rust' in wip


def test_zero_spark_card_sets_build_coordinator_only(tmp_path, monkeypatch, capsys):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('wip_cards_fast0', ROOT / 'scripts/bench/wip-cards.py')
    cards = importlib.util.module_from_spec(spec)
    sys.modules['wip_cards_fast0'] = cards
    spec.loader.exec_module(cards)
    monkeypatch.setattr(cards, 'build_lock_mode', lambda rev: 'native-phase')
    monkeypatch.setattr(cards, 'toolchain_hash', lambda rev: '0123456789abcdef')
    monkeypatch.setattr(cards, 'build_slot_capable', lambda rev: False)
    config = tmp_path / 'card.config'
    config.write_text('SPARK_COUNT=0\n')
    entry = {'name': 'glmf-max', 'family': 'glm5_flash', 'config': str(config), 'sparks': [], 'gpus': [0, 1]}
    arms = cards.arms_from(['cand=c:s'])
    cards.build_arms(['cand=new'], arms, [entry], tmp_path, 'task', True)
    assert '--role coordinator' in capsys.readouterr().out
    assert arms['cand']['build_role'] == 'coordinator'


def test_seed_lease_holds_the_real_flock_until_released(tmp_path):
    """The lease runs through the real hwlock queue: the flock is held while the
    leg runs and free again after release."""
    text = (ROOT / 'wip.sh').read_text()
    helpers = text[text.index('wip_hwlock() {'):text.index('\n}\n', text.index('wip_hwlock() {')) + 3]
    helpers += text[text.index("# The seed Spark's lock: rhea, moa and the out-of-pool build seed kakapo"):text.index('build_coordinator() {')]
    locks = tmp_path / 'locks'
    locks.mkdir()
    probe = 'import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)'
    command = helpers + f'''
repo_root={str(ROOT)!r}
wip_seed_lease_acquire moa
[[ "$CUTEAFD_WIP_SEED_LEASE" == moa ]]
if python3 -c {probe!r} {str(locks / 'moa.lock')!r} 2>/dev/null; then echo free-while-held; fi
wip_seed_lease_release
python3 -c {probe!r} {str(locks / 'moa.lock')!r} && echo free-after
[[ "$(wip_seed_lock rhea) $(wip_seed_lock ostrich) $(wip_seed_lock kakapo)" == "rhea sparks kakapo" ]]
'''
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + command], capture_output=True, text=True, timeout=60,
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(locks), 'TMPDIR': str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert 'free-while-held' not in result.stdout and 'free-after' in result.stdout
