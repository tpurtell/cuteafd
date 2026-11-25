"""Publish only validated original-MTP repair deltas, reusing writer SHA receipts.

No inherited checkpoint payloads are opened, uploaded or hashed. Publication is
parent-guarded HTTP LFS batch commits (metadata last); verification reads pinned
Hub tree metadata only, and resume never resends a committed delta file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re

import hub_lfs
from hub_requirements import require_qualified_hub
from kimi_k3.uniform_k2.storage import atomic_json, file_identity

PARENTS = {
    'wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC': 'c39486376dee84e5faa4904e014e07d7b6e76405',
    'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC': '91a8dbc9c12e790c65aa1a38c6aede6a3e23a77d',
    'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-MC': 'b59496510b33f27d92516e8f4f27e733db4d816e',
    'wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC': '4483a4bd874e8c360aec486ea72bd31d1ad7a247',
}
METADATA = {'model.safetensors.index.json', 'quantize_config.json', 'config.json',
            'cuteafd-mc.json', 'README.md'}
REPORT = 'original-mtp-repair.json'
MAX_METADATA_BYTES = 64 << 20


def field(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('repair delta file must be regular without symlink parents')
    return path


def metadata(path):
    path = regular(path)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError('repair metadata exceeds bounded size')
    return path.read_bytes()


def load(path):
    return json.loads(metadata(path))


def check_sources(files):
    for row in files.values():
        path = regular(row['path'])
        if file_identity(path) != row['file_identity'] or path.stat().st_dev != row['device']:
            raise ValueError('repair delta source changed since receipt')


def delta_packet(delta):
    """Bind only newly written files; reuse shard SHA256, hash small metadata."""
    delta = Path(delta).absolute()
    report = load(delta / REPORT)
    identity = report['identity']
    repo, parent = identity['repo'], identity['parent_revision']
    if (PARENTS.get(repo) != parent or identity.get('schema') != 'cuteafd-original-mtp-repair-v1'
            or report.get('status') != 'validated-delta-headers-and-streamed-source-bytes'
            or report.get('compact_metadata_unchanged') is not True):
        raise ValueError('matching validated original-MTP delta required')
    shards = report['new_shards']
    if (not shards or any(not re.fullmatch(r'original-mtp-[0-9]{4}\.safetensors', n) for n in shards)
            or set(report['new_files']) != METADATA | set(shards)):
        raise ValueError('repair delta allowlist differs from validated metadata and new shards')
    if {p.name for p in delta.iterdir()} != METADATA | set(shards) | {REPORT}:
        raise ValueError('repair delta contains unreviewed files')
    files = {}
    for name in sorted(METADATA | set(shards) | {REPORT}):
        path = regular(delta / name)
        md = file_identity(path)
        if name != REPORT:
            row = report['new_files'][name]
            if md != row['file_identity'] or md['bytes'] != row['bytes']:
                raise ValueError('repair file changed since preparation')
        if name in shards:
            writer = shards[name]
            digest = writer['sha256']
            if md != writer['file_identity'] or digest != row['sha256']:
                raise ValueError('new shard writer receipt differs')
            git_blob = None
        else:
            raw = metadata(path)
            digest = hashlib.sha256(raw).hexdigest()
            if name != REPORT and digest != row['sha256']:
                raise ValueError('repair metadata digest differs')
            git_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid repair writer digest')
        files[name] = dict(path=str(path), bytes=md['bytes'], sha256=digest,
                           git_blob=git_blob, file_identity=md, device=path.stat().st_dev)
    return dict(schema='cuteafd-original-mtp-delta-publication-v1', identity=identity,
                repo_id=repo, parent_revision=parent, delta=str(delta), files=files)


def hub_tree(api, repo, revision):
    """Read names, sizes and Git/LFS OIDs; never download remote weights."""
    if not re.fullmatch('[0-9a-f]{40,64}', revision or ''):
        raise ValueError('pinned Hub revision required')
    info = api.model_info(repo, revision=revision)
    if field(info, 'sha') != revision or field(info, 'private') is not False:
        raise ValueError('repair publication must remain public at pinned revision')
    files = {}
    for row in api.list_repo_tree(repo, revision=revision, recursive=True):
        blob = field(row, 'blob_id')
        if blob is None:  # RepoFolder, not a file.
            continue
        name, size, lfs = field(row, 'path'), field(row, 'size'), field(row, 'lfs')
        if (not isinstance(name, str) or name.startswith('/') or '\\' in name
                or any(p in ('', '.', '..') for p in name.split('/')) or name in files
                or type(size) is not int or size < 0):
            raise ValueError('invalid or duplicate Hub tree file')
        oid = field(lfs, 'sha256') if lfs else blob
        if not re.fullmatch('[0-9a-f]{64}' if lfs else '[0-9a-f]{40}', oid or ''):
            raise ValueError('Hub tree file lacks payload identity')
        files[name] = dict(bytes=size, oid=oid, storage='lfs' if lfs else 'git')
    if not files:
        raise ValueError('empty pinned parent tree')
    return files


def verify_tree(actual, parent, changes, normalized_attributes=None):
    if set(actual) != set(parent) | set(changes):
        raise ValueError('repair publication tree inventory differs')
    for name, row in actual.items():
        if name not in changes:
            expected = normalized_attributes if name == '.gitattributes' and normalized_attributes else parent[name]
            if row != expected:
                raise ValueError('inherited Hub file changed: ' + name)
        else:
            expected = changes[name]
            digest = expected['sha256'] if row['storage'] == 'lfs' else expected['git_blob']
            if row['bytes'] != expected['bytes'] or row['oid'] != digest:
                raise ValueError('published repair file identity differs: ' + name)


def committed_changes(actual, parent, changes):
    """Delta files an interrupted publication committed; anything else refuses."""
    done = set()
    for name, row in actual.items():
        if name == '.gitattributes':
            continue  # Checked against literal appended rules by normalized_attributes.
        if name in changes:
            expected = changes[name]
            digest = expected['sha256'] if row['storage'] == 'lfs' else expected['git_blob']
            if row['bytes'] == expected['bytes'] and row['oid'] == digest:
                done.add(name)
                continue
        if parent.get(name) != row:
            raise ValueError('repair publication tree differs from parent and delta: ' + name)
    if set(parent) - set(actual):
        raise ValueError('repair publication tree inventory differs')
    return done


def normalized_attributes(actual, parent, changes, revision, root, fetch):
    """Permit only literal Hub-appended LFS rules for changed repair files."""
    name = '.gitattributes'
    if actual.get(name) == parent.get(name):
        return None
    before, after = parent.get(name), actual.get(name)
    shards = {n for n in changes if actual.get(n, {}).get('storage') == 'lfs'}
    if (not before or not after or before['storage'] != 'git' or after['storage'] != 'git'
            or not 0 < before['bytes'] < after['bytes'] <= before['bytes'] + 4096 * len(shards)
            or after['bytes'] > 1 << 20):
        raise ValueError('unexpected inherited attributes change')
    old = fetch('parent', name)
    new = fetch(revision, name)
    for raw, row in ((old, before), (new, after)):
        if (not isinstance(raw, bytes) or len(raw) != row['bytes']
                or hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != row['oid']):
            raise ValueError('attributes metadata differs from pinned Git identity')
    if not old.endswith(b'\n') or not new.startswith(old):
        raise ValueError('unexpected inherited attributes rewrite')
    added = new[len(old):].splitlines(keepends=True)
    allowed = {n.encode() + b' filter=lfs diff=lfs merge=lfs -text\n' for n in shards}
    if not added or len(added) != len(set(added)) or not set(added) <= allowed:
        raise ValueError('unexpected inherited attributes rules')
    target = root / 'published.gitattributes'
    # Batches only append literal rules, so a recorded copy may only grow.
    if target.is_symlink() or (target.exists() and not new.startswith(metadata(target))):
        raise ValueError('normalized attributes recovery file differs')
    if not target.exists() or metadata(target) != new:
        target.write_bytes(new)
    return after


def publish(delta, root, *, api=None, anonymous_api=None, fetch_metadata=None, options=None):
    """Commit a delta in verified batches and resume from exact remote metadata."""
    from huggingface_hub import HfApi, hf_hub_download

    require_qualified_hub()
    options = options or hub_lfs.UploadOptions()
    root, delta = Path(root).absolute(), Path(delta).absolute()
    if root == delta or root in delta.parents or delta in root.parents:
        raise ValueError('repair delta and publication receipts must be disjoint')
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('repair publication root cannot traverse symlinks')
    root.mkdir(parents=True, exist_ok=True)
    api, anonymous_api = api or HfApi(), anonymous_api or HfApi(token=False)
    with (root / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path, marker = root / 'publication-plan.json', root / 'publication-complete.json'
        if marker.exists():
            saved, plan = load(marker), load(plan_path)
            if saved.get('status') != 'passed' or saved.get('identity') != plan['identity'] or saved.get('delta') != str(delta):
                raise ValueError('repair completion receipt differs')
            verify_tree(hub_tree(anonymous_api, plan['repo_id'], saved['revision']),
                        saved['parent_files'], plan['files'], saved.get('normalized_attributes'))
            return saved  # A completed publication needs no surviving delta payload.
        plan = delta_packet(delta)
        if plan_path.exists() and load(plan_path) != plan:
            raise ValueError('repair recovery plan differs')
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        repo, parent = plan['repo_id'], plan['parent_revision']
        download = fetch_metadata or (lambda revision, name: Path(hf_hub_download(repo, name,
            revision=parent if revision == 'parent' else revision, token=getattr(api, 'token', False),
            local_dir=root / 'hub-metadata' / revision)).read_bytes())
        fetched = {}

        def fetch(revision, name):
            # Pinned revisions are immutable: per-batch verification fetches each once.
            if (revision, name) not in fetched:
                fetched[revision, name] = download(revision, name)
            return fetched[revision, name]

        parent_files = hub_tree(api, repo, parent)
        if set(plan['files']) & set(parent_files) != METADATA:
            raise ValueError('new repair payload collides with pinned parent')
        check_sources(plan['files'])
        head = api.model_info(repo)
        if field(head, 'private') is not False:
            raise ValueError('repair destination became private')
        revision = field(head, 'sha')
        attempt = root / 'publication-attempt.json'
        changes = plan['files']
        done = set()
        if revision != parent:
            if not attempt.exists() or load(attempt) != dict(identity=plan['identity'], parent_revision=parent):
                raise ValueError('Hub head moved before guarded repair commit')
            shown = hub_tree(api, repo, revision)
            done = committed_changes(shown, parent_files, changes)
            committed = {name: changes[name] for name in done}
            verify_tree(shown, parent_files, committed,
                        normalized_attributes(shown, parent_files, committed, revision, root, fetch))
        else:
            atomic_json(attempt, dict(identity=plan['identity'], parent_revision=parent))
        pending = [name for name in hub_lfs.upload_order(changes) if name not in done]

        def verify(commit, batch, operations):
            done.update(batch)
            shown = hub_tree(api, repo, commit)
            committed = {name: changes[name] for name in done}
            verify_tree(shown, parent_files, committed,
                        normalized_attributes(shown, parent_files, committed, commit, root, fetch))

        if pending:
            revision = hub_lfs.commit_batches(api, repo, changes, pending, revision=revision,
                commit_message='Restore original MTP core in default MC view', options=options,
                check_sources=lambda batch: check_sources({name: changes[name] for name in batch}),
                verify=verify)
        check_sources(plan['files'])
        actual = hub_tree(api, repo, revision)
        normalized = normalized_attributes(actual, parent_files, plan['files'], revision, root, fetch)
        verify_tree(actual, parent_files, plan['files'], normalized)
        verify_tree(hub_tree(anonymous_api, repo, revision), parent_files, plan['files'], normalized)
        record = dict(schema='cuteafd-original-mtp-publication-complete-v1', status='passed',
            identity=plan['identity'], repo_id=repo, parent_revision=parent, revision=revision,
            delta=str(delta), url=f'https://huggingface.co/{repo}/tree/{revision}',
            parent_files=parent_files, files=actual, changed_files=plan['files'],
            normalized_attributes=normalized,
            verification='Exact pinned Hub tree LFS/Git identities and sizes; no remote weight reads',
            time=datetime.now(timezone.utc).isoformat())
        atomic_json(marker, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    hub_lfs.add_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(publish(args.delta, args.root, options=hub_lfs.options_from(args))), flush=True)


if __name__ == '__main__':
    main()
