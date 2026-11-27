"""Content-keyed WIP reuse: sealed slot adoption, native-output cache, fast source checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

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
    assert reuse(tmp_path, 'claim', '--root', str(root), '--slot', 'gate', '--generation', 'fixture').returncode == 0
    adopted = reuse(tmp_path, 'adopt', '--key', KEY, '--role', 'coordinator', '--source', str(source),
                    '--root', str(root), '--slot', 'gate', '--generation', 'fixture', '--instance', 'mine', '--wip-root', str(root))
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
    assert reuse(tmp_path, 'claim', '--root', str(root), '--slot', 'gate', '--generation', 'fixture').returncode == 0
    adopted = reuse(tmp_path, 'adopt', '--key', KEY, '--role', 'coordinator', '--source', str(own),
                    '--root', str(root), '--slot', 'gate', '--generation', 'fixture', '--instance', 'mine')
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
                  image_id='id', image_layers='["sha256:l1","sha256:l2"]',
                  source_manifest_sha256='s', artifact_sha256='a', sparkinfer_revision='r',
                  transformers_revision='t', transformers_source_digest='d', build_output=str(output),
                  incoming=str(incoming))
    prefix = ''.join(f'{k}={v!r}\n' for k, v in values.items())
    subprocess.run(['bash', '-c', 'set -euo pipefail\n' + prefix + block], check=True)
    meta = json.loads((incoming / 'META.json').read_text())
    assert meta['build_key'] == KEY
    assert meta['base_image_layers'] == ['sha256:l1', 'sha256:l2']
    assert [(p['phase'], p['seconds']) for p in meta['build_phases']] == [('rust', 12), ('native', 3)]


def _wip_coordinator(tmp_path, hit, exports_cached=False, incomplete=False):
    text = (ROOT / 'wip.sh').read_text()
    helpers = ''
    helpers += text[text.index('# Native/AOT outputs are cached by input key'):text.index('build_expert() (')]
    tools = tmp_path / 'tools'
    tools.mkdir()
    trace = tmp_path / 'trace'
    cached = 'true' if exports_cached else 'false'
    for name, body in {'docker': 'printf "docker %s\\n" "$*" >> "$TRACE"\nprintf "image-id\\n"\n'
                                 'case "$*" in *CUTEAFD_EXPORT_CACHE_REQUIRE_HIT=1*) ' + cached + ' || exit 3;; esac',
                       'python3': 'case "$1" in */build-process.py) exec ' + sys.executable + ' "$@";; esac\n'
                                  'printf "turn %s\\n" "$*" >> "$TRACE"\ncase "$*" in *waiting-serving*) exit 1;; esac'}.items():
        (tools / name).write_text('#!/bin/sh\n' + body + '\n')
        (tools / name).chmod(0o755)
    state = tmp_path / 'state'
    (state / 'output/coordinator').mkdir(parents=True)
    (state / 'output/coordinator/.native-cache-key').write_text('coordinator-120 ' + KEY + '\n')
    if hit:
        required = native_inventory(tmp_path)
        (state / 'output/coordinator/.native-cache-required').write_bytes(required.read_bytes())
        entry = native_entry(tmp_path, required)
        cached_entry = state / 'cache/native-store/coordinator-120' / KEY
        cached_entry.parent.mkdir(parents=True)
        entry.rename(cached_entry)
        if incomplete:
            (cached_entry / 'output/exl3/exl3-qwen4-k4/manifest.json').unlink()
            seal_native_entry(cached_entry)
    setup = f'''set -euo pipefail
repo_root={str(ROOT)!r}
state_dir={str(state)!r}
sync_local_source() {{ :; }}
release_die() {{ exit 2; }}
cuteafd_compiler_cache_warn() {{ :; }}
slot=test coordinator_container=owned COORDINATOR_DOCKER_DEV=image bf16_families= audio_aot=OFF
'''
    result = subprocess.run(['bash', '-c', setup + helpers + '\nbuild_coordinator all "$state_dir/source-staging"'], capture_output=True, text=True,
                            env={**os.environ, 'HOME': str(tmp_path / 'home'), 'PATH': f'{tools}:{os.environ["PATH"]}',
                                 'TRACE': str(trace)})
    assert result.returncode == 0, result.stderr
    return result.stdout, trace.read_text().splitlines()


def test_native_cache_hit_takes_no_gpu_turn(tmp_path):
    stdout, lines = _wip_coordinator(tmp_path, hit=True)
    assert 'WIP coordinator export: native cache hit' in stdout
    assert not any(line.startswith('turn ') for line in lines)
    native = [line for line in lines if line.endswith('/wip/output/coordinator native')]
    assert len(native) == 1 and 'CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1' in native[0]


def test_incomplete_native_cache_entry_falls_through_to_rebuild(tmp_path):
    stdout, lines = _wip_coordinator(tmp_path, hit=True, incomplete=True)
    assert 'WIP coordinator export: native cache hit' not in stdout
    assert 'WIP coordinator export: CPU export' in stdout
    assert not any('CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1' in line for line in lines)


def test_native_cache_miss_runs_cpu_export_without_gpu_lock(tmp_path):
    stdout, lines = _wip_coordinator(tmp_path, hit=False)
    assert 'WIP coordinator export: CPU export' in stdout
    native = [line for line in lines if line.endswith('/wip/output/coordinator native')]
    assert len(native) == 1 and 'CUDA_VISIBLE_DEVICES=' in native[0]
    assert not any('hwlock.py --locks gpu1' in line for line in lines)


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
                 'python3 "$source_dir/scripts/build/native-key-files.py" "$source_dir"', '"$role" "$cuda_arch"'):
        assert part in block, part
    assert "TREES = ('native', 'python', 'scripts/build')" in (ROOT / 'scripts/build/native-key-files.py').read_text()
    # A hit restores every native output and nothing else; the daemon always comes from Cargo.
    assert 'native_items=(libcuteafd_native.so lib-variants PROGRAMS.json V41_EXPERT_AOT.json V41_EXPERT_TP_AOT.json V41_FP8_AOT.json V41_EXPERT_WORKSPACES exl3 fp8)' in script
    assert script.index('cp -a "$native_entry/output/." "$output_dir/"') < script.index('install -m 0755 "$(cuteafd_host_isa_daemon "$CARGO_TARGET_DIR" "${host_isas[0]}")"')


def native_inventory(tmp_path, *, families='qwen4:exl3-k4;mimo:fp8', role='coordinator',
                     bf16='', v41='ON', nvfp4='ON'):
    required = tmp_path / '.native-cache-required'
    command = '''set -euo pipefail
source "$1/scripts/build/wip-native-store.sh"
role="$2"; expert_families="$3"; bf16_families="$4"
v41_coordinator_aot="$5"; v41_nvfp4_aot="$6"; exl3_aot=ON
host_isas=(x86-64-v4 x86-64-v2)
wip_native_required_items
'''
    result = subprocess.run(['bash', '-c', command, 'test', str(ROOT), role, families, bf16, v41, nvfp4],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    required.write_text(result.stdout)
    return required


def native_entry(tmp_path, required):
    entry = tmp_path / 'entry'
    output = entry / 'output'
    output.mkdir(parents=True)
    for line in required.read_text().splitlines():
        kind, name = line.split(' ', 1)
        path = output / name
        if kind == 'dir':
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
    (entry / 'COMPLETE').write_text(KEY + '\n')
    seal_native_entry(entry)
    return entry


def seal_native_entry(entry):
    output = entry / 'output'
    (entry / 'NATIVE_SHA256SUMS').write_text(''.join(
        f'{sha(path.read_bytes())}  ./{path.relative_to(output)}\n'
        for path in sorted(output.rglob('*')) if path.is_file()))


def validate_native_entry(entry, required):
    return subprocess.run(['bash', '-c', 'set -euo pipefail\nsource "$1"\n'
                           'wip_native_store_valid "$2" "$3"', 'test',
                           str(ROOT / 'scripts/build/wip-native-store.sh'), str(entry), str(required)],
                          capture_output=True, text=True, timeout=30)


def test_complete_native_store_entry_is_accepted(tmp_path):
    required = native_inventory(tmp_path)
    entry = native_entry(tmp_path, required)
    assert validate_native_entry(entry, required).returncode == 0


@pytest.mark.parametrize('missing', ['exl3', 'fp8', 'PROGRAMS.json', 'V41_EXPERT_AOT.json',
                                    'V41_EXPERT_TP_AOT.json', 'V41_FP8_AOT.json',
                                    'V41_EXPERT_WORKSPACES', 'lib-variants',
                                    'exl3/exl3-qwen4-k4/manifest.json',
                                    'fp8/fp8-mimo/manifest.json',
                                    'V41_EXPERT_WORKSPACES/mxfp4_rtx_tp2.json',
                                    'V41_EXPERT_WORKSPACES/mxfp4_dspark_tp2.json',
                                    'lib-variants/x86-64-v2/libcuteafd_native.so'])
def test_native_store_rejects_scope_limited_or_incomplete_entry(tmp_path, missing):
    import shutil
    required = native_inventory(tmp_path)
    entry = native_entry(tmp_path, required)
    absent = entry / 'output' / missing
    shutil.rmtree(absent) if absent.is_dir() else absent.unlink()
    # The partial entry's checksums are internally valid: completeness must be
    # checked against this build's inventory, not just the stored file list.
    seal_native_entry(entry)
    result = validate_native_entry(entry, required)
    assert result.returncode == 1 and 'missing required' in result.stderr
    assert (entry / 'COMPLETE').is_file()  # refusal never removes old entries


def test_native_store_rejects_corrupt_payload(tmp_path):
    required = native_inventory(tmp_path)
    entry = native_entry(tmp_path, required)
    (entry / 'output/libcuteafd_native.so').write_bytes(b'corrupt')
    result = validate_native_entry(entry, required)
    assert result.returncode == 1 and 'native checksums failed' in result.stderr


def test_limited_native_scope_does_not_require_unrequested_packages(tmp_path):
    required = native_inventory(tmp_path, families='', v41='OFF', nvfp4='OFF')
    entry = native_entry(tmp_path, required)
    assert not (entry / 'output/exl3').exists()
    assert not (entry / 'output/fp8').exists()
    assert validate_native_entry(entry, required).returncode == 0


@pytest.mark.parametrize('families,role,bf16,packages', [
    ('mimo:fp8', 'expert', 'mimo', ['fp8-mimo', 'fp8-mimo-bf16']),
    ('qwen4:nvfp4', 'coordinator', '', ['fp8-qwen4-nvfp4', 'fp8-qwen4-nvfp4a4', 'fp8-qwen4']),
    ('step5:bf16;step5:nvfp4', 'expert', '', ['fp8-step5-bf16-weights', 'fp8-step5-nvfp4a4']),
])
def test_native_inventory_requires_every_staged_package_sibling(tmp_path, families, role, bf16, packages):
    required = native_inventory(tmp_path, families=families, role=role, bf16=bf16)
    manifests = [line.split('/')[1] for line in required.read_text().splitlines() if line.startswith('file fp8/')]
    assert manifests == packages


def test_host_native_probe_uses_the_same_validation_before_skipping_export(tmp_path):
    required = native_inventory(tmp_path)
    entry = native_entry(tmp_path, required)
    root = tmp_path / 'instance'
    output = root / 'output/coordinator'
    output.mkdir(parents=True)
    (output / '.native-cache-key').write_text(f'coordinator-120 {KEY}\n')
    (output / '.native-cache-required').write_bytes(required.read_bytes())
    cached = root / 'cache/native-store/coordinator-120' / KEY
    cached.parent.mkdir(parents=True)
    entry.rename(cached)
    script = (ROOT / 'wip.sh').read_text()
    start = script.index('native_store_has() {')
    function = script[start:script.index('\n}\n', start) + 3]
    command = 'set -euo pipefail\nrepo_root="$1"\n' + function + '\nnative_store_has "" "$2" coordinator'
    args = ['bash', '-c', command, 'test', str(ROOT), str(root)]
    assert subprocess.run(args, capture_output=True, timeout=30).returncode == 0
    (cached / 'output/exl3/exl3-qwen4-k4/manifest.json').unlink()
    seal_native_entry(cached)
    rejected = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert rejected.returncode == 1 and 'missing required' in rejected.stderr


def test_source_checkout_initializes_only_build_submodules_from_local_stores():
    text = (ROOT / 'scripts/build/wip-source-checkout.sh').read_text()
    for name in ('third_party/sparkinfer', 'third_party/transformers', 'third_party/xgrammar'):
        assert f'submodule_from_local "$dest" {name}' in text
    assert 'gptqmodel "' not in text and 'tool-eval-bench "' not in text
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
    cards.build_arms({'cand': 'new'}, arms, [entry], tmp_path, 'task', True)
    text = capsys.readouterr().out
    assert 'build.lock' not in text and 'wip.sh --slot s --role both' in text
    assert arms['cand']['build_slot'] is True
    source = (ROOT / 'scripts/bench/wip-cards.py').read_text()
    # No pre-build seed wait: wip.sh queues each seed and Spark step itself (#2724).
    assert 'CUTEAFD_WIP_SEED_WAIT_S' not in source and "'--', 'true'" not in source
    wip = (ROOT / 'wip.sh').read_text()
    assert 'resource exec --locks build --nice 19 --stop-grace 25' in wip


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
    config.write_text('MODEL_ID=zai-org/GLM-5.3-Flash\nSPARK_COUNT=0\n')
    entry = {'name': 'glmf-max', 'family': 'glm5_flash', 'config': str(config), 'sparks': [], 'gpus': [0, 1]}
    arms = cards.arms_from(['cand=c:s'])
    cards.build_arms({'cand': 'new'}, arms, [entry], tmp_path, 'task', True)
    assert '--role coordinator' in capsys.readouterr().out
    assert arms['cand']['build_role'] == 'coordinator'
    staged = (tmp_path / 'build-cand.config').read_text()
    assert 'SPARK_COUNT=0\n' in staged
    assert 'SPARK_0_HOST' not in staged


def test_seed_uses_only_owner_aware_build_resource():
    text = (ROOT / 'wip.sh').read_text()
    assert 'wip_seed_lease_' not in text
    assert 'wip_build_resource_acquire' in text
    helpers = text[text.index('wip_seed_lock() {'):text.index('\n}\n', text.index('wip_seed_lock() {')) + 3]
    common = (ROOT / 'scripts/lib/release-common.sh').read_text()
    lan = common[common.index('release_lan_seed_hosts=('):common.index('# release_host_has_rdma HOST')]
    result = subprocess.run(['bash', '-c', lan + helpers + '\nfor h in rhea ostrich kakapo cassowary; do wip_seed_lock $h; done'],
                            capture_output=True, text=True, timeout=5)
    assert result.stdout.split() == ['rhea', 'sparks', 'kakapo', 'cassowary']
