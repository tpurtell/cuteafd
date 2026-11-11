"""Post-publish transfer never copies or downloads checkpoint payloads."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc_postpublish as post


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def staged_inputs(tmp_path):
    package = tmp_path / 'package'
    package.mkdir()
    raw = b'checkpoint bytes'
    (package / 'weight.safetensors').write_bytes(raw)
    attrs = tmp_path / 'server-attributes'
    attrs.write_bytes(b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    rows = {'weight.safetensors': dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())}
    rows['.gitattributes'] = dict(bytes=attrs.stat().st_size, sha256=post.sha256_file(attrs))
    packet = dump(tmp_path / 'packet.json', dict(repo_id='owner/model', new_files=rows, copy_files={}, copies={}))
    audit = dump(tmp_path / 'audit.json', dict(repo_id='owner/model', commit='a' * 40,
        status='anonymous-exact-tree-sizes-and-payload-identities-verified', packet_sha256=post.sha256_file(packet),
        files={n: dict(r, storage='lfs' if n.endswith('.safetensors') else 'git') for n, r in rows.items()}))
    return dict(package=package, destination=tmp_path / 'stage', packet_path=packet, audit_path=audit,
                attributes=attrs, receipt_path=tmp_path / 'stage-receipt.json')


def test_stage_is_hardlinks_and_standard_cache_layout(staged_inputs):
    result = post.stage(**staged_inputs)
    repo = Path(result['repo'])
    path = repo / 'snapshots' / ('a' * 40) / 'weight.safetensors'
    assert path.is_symlink()
    assert path.stat().st_ino == (staged_inputs['package'] / path.name).stat().st_ino
    assert (repo / 'refs/main').read_text() == 'a' * 40
    assert result['copied_bytes'] == result['fetched_bytes'] == 0
    assert not (repo / 'snapshots' / ('a' * 40) / 'stage-receipt.json').exists()


def test_missing_or_changed_payload_fails_before_creating_stage(staged_inputs):
    (staged_inputs['package'] / 'weight.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='differs'):
        post.stage(**staged_inputs)
    assert not staged_inputs['destination'].exists()


def test_hardlink_failure_never_falls_back_to_copy(staged_inputs, monkeypatch):
    def fail(*args):
        raise OSError(18, 'cross-device link')
    monkeypatch.setattr(os, 'link', fail)
    with pytest.raises(OSError):
        post.stage(**staged_inputs)
    assert not staged_inputs['receipt_path'].exists()


def test_invalid_published_path_rejected(staged_inputs):
    packet = post.load(staged_inputs['packet_path'])
    packet['new_files']['../escape'] = packet['new_files']['weight.safetensors']
    dump(staged_inputs['packet_path'], packet)
    audit = post.load(staged_inputs['audit_path'])
    audit['packet_sha256'] = post.sha256_file(staged_inputs['packet_path'])
    audit['files']['../escape'] = audit['files']['weight.safetensors']
    dump(staged_inputs['audit_path'], audit)
    with pytest.raises(ValueError, match='invalid published'):
        post.stage(**staged_inputs)


@pytest.mark.parametrize('copied,errors', [(0, []), (1, []), (0, ['failed'])])
def test_import_hardcodes_offline_no_copy_and_checks_zero_bytes(staged_inputs, monkeypatch, copied, errors):
    post.stage(**staged_inputs)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        result = dict(job=123) if len(calls) == 1 else dict(
            job=dict(finished=True, cancelled=False, error=None),
            **{'import': dict(copied_bytes=copied, errors=errors, cancelled=False)})
        return SimpleNamespace(stdout=json.dumps(result))
    monkeypatch.setattr(post.subprocess, 'run', run)
    receipt = staged_inputs['receipt_path'].parent / 'import-receipt.json'
    if copied or errors:
        with pytest.raises(ValueError, match='did not complete'):
            post.import_staged(staged_inputs['receipt_path'], receipt)
    else:
        result = post.import_staged(staged_inputs['receipt_path'], receipt)
        assert result['status'] == 'offline-no-copy-imported'
        assert result['fetched_bytes'] == result['copied_bytes'] == 0
    assert '--offline' in calls[0] and '--no-copy' in calls[0]
    assert '--copy' not in calls[0] and '--spread' not in calls[0]


def test_cleanup_never_accepts_unverified_receipt(tmp_path):
    receipt = dump(tmp_path / 'verify.json', dict(status='verification-running'))
    with pytest.raises(ValueError, match='verified imported'):
        post.cleanup(receipt, [tmp_path], tmp_path / 'clean.json', [])


def test_cleanup_rejects_shared_stores_and_protected_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    builds = tmp_path / '.cache/cuteafd/builds'
    builds.mkdir(parents=True)
    receipt = dump(tmp_path / 'verify.json', dict(status='imported-hashes-and-readiness-verified', snapshot='/mnt/sparknest/model'))
    for root, protected in [(builds, []), (tmp_path / '.cache/cuteafd/instances', []), (builds / 'active', [builds / 'active/input'])]:
        with pytest.raises(ValueError, match='unsafe or protected'):
            post.cleanup(receipt, [root], tmp_path / 'clean.json', protected)


def test_inherited_alias_is_same_inode_without_payload_copy(staged_inputs, tmp_path):
    packet = post.load(staged_inputs['packet_path'])
    row = packet['new_files'].pop('weight.safetensors')
    packet['copy_files']['weight.safetensors'] = row
    packet['copies']['weight.safetensors'] = 'parent.safetensors'
    dump(staged_inputs['packet_path'], packet)
    audit = post.load(staged_inputs['audit_path'])
    audit['packet_sha256'] = post.sha256_file(staged_inputs['packet_path'])
    dump(staged_inputs['audit_path'], audit)
    hub = tmp_path / 'hub'
    source = hub / 'parent/snapshots/revision/parent.safetensors'
    source.parent.mkdir(parents=True)
    source.write_bytes((staged_inputs['package'] / 'weight.safetensors').read_bytes())
    inherited = {'weight.safetensors': dict(source=str(source), oid=row['sha256'], **row)}
    staged = post.stage(**staged_inputs, inherited=inherited)
    assert not (Path(staged['repo']) / 'snapshots' / ('a' * 40) / 'weight.safetensors').exists()
    receipt = tmp_path / 'alias.json'
    result = post.alias_inherited(staged_inputs['receipt_path'], hub, receipt, limit=1)
    target = hub / 'models--owner--model/snapshots' / ('a' * 40) / 'weight.safetensors'
    assert target.is_symlink() and target.stat().st_ino == source.stat().st_ino
    assert source.stat().st_nlink == 2
    assert result['copied_bytes'] == result['fetched_bytes'] == 0
    post.alias_inherited(staged_inputs['receipt_path'], hub, receipt)
    assert source.stat().st_nlink == 2
