"""Publish one completed MC directory, checking Hub metadata without rereading weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kimi_k3.uniform_k2.storage import atomic_json, canonical, file_identity, verify_receipt

TARGET_REPO = 'wrldsuksgo2mars/Inkling-2-Small-EXL3-K4-MC'
MAX_METADATA_BYTES = 64 << 20


def load(path):
    return json.loads(Path(path).read_text())


def relative(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or p.as_posix() != name
            or any(v in ('.', '..') for v in p.parts) or any(c in name for c in '\\\0\r\n')):
        raise ValueError('unsafe package relative path')
    return p


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('publication file must be regular without symlink parents')
    return path


def metadata_identity(path):
    path = regular(path)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError('non-weight metadata exceeds bounded size')
    raw = path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(),
                git_blob=hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest())


def manifest_digest(files):
    return hashlib.sha256(canonical({n: {k: v[k] for k in ('bytes', 'sha256')}
                                     for n, v in sorted(files.items())}).encode()).hexdigest()


def publication_inventory(output):
    """Use validated streaming receipts for weights; read only small metadata."""
    output = Path(output).resolve(strict=True)
    state, validation = (load(output / n) for n in ('package-state.json', 'package-validation.json'))
    if (state.get('identity') != validation.get('identity') or state.get('copies') != {}
            or state['identity'].get('family') != 'inkling2' or 'direct_core_selection' not in state
            or validation.get('status') != 'validated-headers-and-byte-exactness'):
        raise ValueError('publication requires validated independent Inkling MC package')
    names = set(state['new_files']) | {'package-validation.json', 'package-provenance.json'}
    required = {'README.md', '.gitattributes', 'cuteafd-mc.json', 'config.json', 'config.k8.json',
        'config.k6.json', 'quantize_config.json', 'quantize_config.k8.json', 'quantize_config.k6.json',
        'model.safetensors.index.json', 'model.k8.safetensors.index.json', 'model.k6.safetensors.index.json'}
    if not required <= names or {p.name for p in output.glob('*.safetensors')} != set(state['new_shards']):
        raise ValueError('publication inventory lacks metadata or has unowned weights')
    files = {}
    for name in sorted(names):
        relative(name)
        path = regular(output / name)
        identity = file_identity(path)
        if name.endswith('.safetensors'):
            receipt = state['new_shards'][name]
            verify_receipt(path, receipt)  # stat only: the writer already hashed its byte stream.
            digest = receipt['sha256']
            if validation['new_file_sha256'].get(name) != digest:
                raise ValueError('validated shard SHA differs from writer receipt')
            hashes = dict(sha256=digest, git_blob=None)
        else:
            hashes = metadata_identity(path)
            if name in validation['new_file_sha256'] and validation['new_file_sha256'][name] != hashes['sha256']:
                raise ValueError('validated metadata changed: ' + name)
        if not re.fullmatch('[0-9a-f]{64}', hashes['sha256']):
            raise ValueError('invalid writer SHA receipt')
        files[name] = dict(path=str(path), bytes=identity['bytes'], file_identity=identity,
                           device=path.stat().st_dev, **hashes)
    return dict(schema='cuteafd-inkling2-publication-plan-v1', repo_id=TARGET_REPO,
        output=str(output), identity=state['identity'], source_snapshot=state['source_snapshot'],
        files=files, manifest_sha256=manifest_digest(files))


def _field(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def verify_hub(api, revision, files, *, partial=False):
    """Compare every pinned Hub LFS SHA256/Git blob and size; fetch no payloads."""
    if not re.fullmatch('[0-9a-f]{40,64}', revision or ''):
        raise ValueError('immutable Hub revision required')
    info = api.model_info(TARGET_REPO, revision=revision, files_metadata=True)
    if _field(info, 'sha') != revision or _field(info, 'private') is not False:
        raise ValueError('published repository must be public at the pinned revision')
    siblings = _field(info, 'siblings') or []
    remote = {_field(row, 'rfilename'): row for row in siblings}
    if len(remote) != len(siblings) or (not partial and set(remote) != set(files)) or set(remote) - set(files):
        raise ValueError('Hub file inventory differs from publication')
    for name, row in remote.items():
        expected = files[name]
        lfs = _field(row, 'lfs')
        if (_field(row, 'size') != expected['bytes']
                or (lfs is not None and _field(lfs, 'sha256') != expected['sha256'])
                or (lfs is None and (name.endswith('.safetensors')
                    or _field(row, 'blob_id') != expected['git_blob']))):
            raise ValueError('Hub file identity differs: ' + name)
    return info


def _attribute_files(api, revision, files, root, fetch_metadata):
    """Allow only Hub-appended literal LFS rules; read that tiny metadata file."""
    info = api.model_info(TARGET_REPO, revision=revision, files_metadata=True)
    remote = {_field(v, 'rfilename'): v for v in (_field(info, 'siblings') or [])}
    name = '.gitattributes'
    row = remote.get(name)
    if row is None or (_field(row, 'size') == files[name]['bytes']
            and _field(row, 'blob_id') == files[name]['git_blob']):
        return files
    original = Path(files[name]['path']).read_bytes()
    size = _field(row, 'size')
    if (not original.endswith(b'\n') or not isinstance(size, int)
            or not len(original) < size <= len(original) + 4096 * len(files)):
        raise ValueError('unexpected server-normalized attributes')
    raw = fetch_metadata(revision, name)
    if not isinstance(raw, bytes) or len(raw) != size or not raw.startswith(original):
        raise ValueError('unexpected server-normalized attributes')
    added = raw[len(original):].splitlines(keepends=True)
    allowed = {n.encode() + b' filter=lfs diff=lfs merge=lfs -text\n'
               for n in files if not any(c in n for c in '*?[]\\\r\n\t ')}
    if not added or len(added) != len(set(added)) or not set(added) <= allowed:
        raise ValueError('unexpected server-normalized attributes')
    target = root / 'published.gitattributes'
    if target.is_symlink():
        raise ValueError('normalized metadata destination is a symlink')
    target.write_bytes(raw)
    hashes = metadata_identity(target)
    result = dict(files)
    result[name] = dict(path=str(target), bytes=len(raw), file_identity=file_identity(target),
                        device=target.stat().st_dev, **hashes)
    return result


def _check_sources(files):
    for row in files.values():
        p = regular(row['path'])
        if file_identity(p) != row['file_identity'] or p.stat().st_dev != row['device']:
            raise ValueError('publication source changed since receipt')


def publish(output, root, *, api=None, anonymous_api=None, fetch_metadata=None):
    """Resume native Hub upload, then seal a metadata-verified publication receipt.

    Call only after numerical completion. The SDK's native upload cache handles
    interrupted uploads; no parent repository or CommitOperationCopy is used.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('publication receipts and output must be disjoint')
    root.mkdir(parents=True, exist_ok=True)
    api = api or HfApi()
    anonymous_api = anonymous_api or HfApi(token=False)
    fetch_metadata = fetch_metadata or (lambda revision, name: Path(hf_hub_download(
        TARGET_REPO, name, revision=revision, token=api.token)).read_bytes())
    with (root / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / 'publication-complete.json'
        if marker.exists():
            saved = load(marker)
            if (saved.get('repo_id') != TARGET_REPO or saved.get('status') != 'passed'
                    or saved.get('output') != str(output) or saved.get('manifest_sha256') != manifest_digest(saved['files'])):
                raise ValueError('publication completion receipt differs')
            verify_hub(anonymous_api, saved['revision'], saved['files'])
            return saved  # Completed jobs stay complete even after standalone cleanup.
        packet = publication_inventory(output)
        plan_path = root / 'publication-plan.json'
        if plan_path.exists() and load(plan_path) != packet:
            raise ValueError('publication recovery plan differs from sealed output')
        atomic_json(plan_path, packet)
        guard = root / 'publication-destination.json'
        if not guard.exists():
            try:
                info = api.model_info(TARGET_REPO, files_metadata=True)
            except RepositoryNotFoundError:
                api.create_repo(TARGET_REPO, repo_type='model', private=False, exist_ok=False)
                info = api.model_info(TARGET_REPO, files_metadata=True)
            if (_field(info, 'private') is not False
                    or {_field(v, 'rfilename') for v in (_field(info, 'siblings') or [])} - {'.gitattributes'}):
                raise ValueError('destination must be new, empty and public')
            atomic_json(guard, dict(repo_id=TARGET_REPO, initial_revision=_field(info, 'sha')))
        elif load(guard).get('repo_id') != TARGET_REPO:
            raise ValueError('destination recovery guard differs')
        # Native resumable upload may have published a subset before interruption.
        head = api.model_info(TARGET_REPO, files_metadata=True)
        if _field(head, 'private') is not False:
            raise ValueError('destination became private')
        current = {_field(v, 'rfilename'): v for v in (_field(head, 'siblings') or [])}
        if set(current) - {'.gitattributes'}:
            partial_files = _attribute_files(api, _field(head, 'sha'), packet['files'], root, fetch_metadata)
            verify_hub(api, _field(head, 'sha'), partial_files, partial=True)
        _check_sources(packet['files'])
        arguments = dict(repo_id=TARGET_REPO, repo_type='model', folder_path=str(output),
                         allow_patterns=sorted(packet['files']))
        if hasattr(api, 'upload_large_folder'):
            api.upload_large_folder(**arguments, num_workers=4, print_report=True, print_report_every=1800)
        else:
            api.upload_folder(**arguments, commit_message='Publish calibrated Inkling 2 K4 MC checkpoint')
        _check_sources(packet['files'])
        head = api.model_info(TARGET_REPO, files_metadata=True)
        revision = _field(head, 'sha')
        published_files = _attribute_files(api, revision, packet['files'], root, fetch_metadata)
        verify_hub(api, revision, published_files)
        verify_hub(anonymous_api, revision, published_files)
        record = dict(schema='cuteafd-inkling2-publication-complete-v1', status='passed',
            repo_id=TARGET_REPO, revision=revision, url=f'https://huggingface.co/{TARGET_REPO}/tree/{revision}',
            output=str(output), identity=packet['identity'], source_snapshot=packet['source_snapshot'],
            files=published_files, manifest_sha256=manifest_digest(published_files),
            verification='Hub metadata LFS SHA256/Git blob and size; no remote weight reads',
            time=datetime.now(timezone.utc).isoformat())
        atomic_json(marker, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    result = publication_inventory(args.output) if args.inventory_only else publish(args.output, args.root)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
