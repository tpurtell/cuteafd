"""Parallel, resumable HTTP LFS publication shared by every quant publisher.

Each batch is one parent-guarded ``create_commit``. The caller checks its
sources before and after the transfer and verifies the new revision against
Hub metadata before the next batch starts, so an interrupted publication
resumes from the verified remote tree and never sends a committed file again.

Hub SDK 1.33.0 behaviour this relies on (the SDK is not patched):
- ``create_commit(num_threads=N)`` preuploads through ``_upload_lfs_files``,
  which maps one file per worker with ``hf_thread_map(max_workers=N)``: up to
  N files of a batch transfer concurrently.
- One file's multipart parts go up serially (``lfs._upload_parts_iteratively``),
  so each file is a single stream and a batch ends when its largest file does.
  ``upload_order`` starts the largest shards first to shorten that tail.
- The SDK hashes only additions without a known SHA256. ``receipt_operation``
  supplies the export's own digest, so publication never rereads weights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from hub_requirements import require_qualified_hub, use_http_lfs

DEFAULT_WORKERS = 16
DEFAULT_BATCH_FILES = 48
MAX_WORKERS = 64
MAX_BATCH_FILES = 256  # One LFS batch-info request (SDK UPLOAD_BATCH_MAX_NUM_FILES).
MAX_GIT_BYTES = 64 << 20  # Larger regular (non-LFS) Git uploads are refused.


class UploadOptions:
    """Concurrent file uploads per commit and files per verified commit."""

    __slots__ = ('workers', 'batch_files')

    def __init__(self, workers=DEFAULT_WORKERS, batch_files=DEFAULT_BATCH_FILES):
        for value, limit in ((workers, MAX_WORKERS), (batch_files, MAX_BATCH_FILES)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError('upload workers/batch files out of range')
        self.workers, self.batch_files = workers, batch_files

    def __repr__(self):
        return f'UploadOptions(workers={self.workers}, batch_files={self.batch_files})'


def _bounded(limit):
    def parse(text):
        value = int(text)
        if not 1 <= value <= limit:
            raise argparse.ArgumentTypeError(f'must be between 1 and {limit}')
        return value
    return parse


def add_arguments(parser, *, batches=True):
    parser.add_argument('--upload-workers', type=_bounded(MAX_WORKERS), default=DEFAULT_WORKERS,
                        help=f'concurrent HTTP LFS file uploads per commit (default {DEFAULT_WORKERS})')
    if batches:
        parser.add_argument('--upload-batch-files', type=_bounded(MAX_BATCH_FILES),
                            default=DEFAULT_BATCH_FILES,
                            help=f'files per verified commit (default {DEFAULT_BATCH_FILES})')


def options_from(args):
    """Upload options from a publisher CLI, which always publishes over HTTP LFS."""
    use_http_lfs()
    return UploadOptions(args.upload_workers, getattr(args, 'upload_batch_files', DEFAULT_BATCH_FILES))


def upload_order(files):
    """'.gitattributes' first, weights largest first, then small metadata."""
    def key(name):
        if name == '.gitattributes':
            return 0, 0, name
        if name.endswith('.safetensors'):
            return 1, -files[name]['bytes'], name
        return 2, 0, name
    return sorted(files, key=key)


def fingerprint(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def git_blob(path):
    """Git blob id of a small file, or None for a file only LFS may carry."""
    size = Path(path).stat().st_size
    if size > MAX_GIT_BYTES:
        return None
    raw = Path(path).read_bytes()
    if len(raw) != size:
        raise ValueError('file changed while hashing: ' + str(path))
    return hashlib.sha1(f'blob {size}\0'.encode() + raw).hexdigest()


def hashed_rows(paths, cache, *, workers=DEFAULT_WORKERS):
    """Hash files that have no writer receipt once, cached by stat identity.

    HTTP LFS needs every SHA256 before transfer, so a publisher whose export
    recorded none pays one read here. A resumed run reuses the cache and
    reads nothing again unless the file's stat identity changed.
    """
    cache = Path(cache)
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    rows, missing = {}, []
    for name, path in sorted(paths.items()):
        local = fingerprint(path)
        row = saved.get(name)
        if row is not None and row.get('local') == local:
            rows[name] = dict(row, path=str(path))
        else:
            missing.append((name, Path(path), local))

    def digest(item):
        name, path, local = item
        with path.open('rb') as stream:
            sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        blob = git_blob(path)
        if fingerprint(path) != local:
            raise ValueError('publication source changed while hashing: ' + name)
        return name, dict(path=str(path), bytes=local['bytes'], sha256=sha256, git_blob=blob, local=local)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(missing) or 1))) as pool:
        rows.update(pool.map(digest, missing))
    if missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=cache.parent, prefix='.' + cache.name + '.')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump({n: {k: v for k, v in r.items() if k != 'path'} for n, r in rows.items()},
                          stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, cache)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return rows


def check_fingerprints(files, names):
    for name in names:
        if fingerprint(files[name]['path']) != files[name]['local']:
            raise ValueError('publication source changed since receipt: ' + name)


def remote_tree(api, repo_id, revision):
    """Pinned Hub tree as {name: {bytes, blob}}; blob is the LFS SHA256 or Git id."""
    from huggingface_hub.hf_api import RepoFile

    return {entry.path: dict(bytes=entry.size, blob=entry.lfs.sha256 if entry.lfs else entry.blob_id)
            for entry in api.list_repo_tree(repo_id, revision=revision, recursive=True, repo_type='model')
            if isinstance(entry, RepoFile)}


def matches(remote, row):
    """A remote {bytes, blob} is this file as LFS (SHA256) or as a regular Git blob."""
    blobs = {row.get('sha256'), row.get('git_blob')} - {None}
    return remote.get('bytes') == row['bytes'] and remote.get('blob') in blobs


def resolved(remote, row, name):
    if not matches(remote, row):
        raise ValueError('published file identity differs: ' + name)
    if name.endswith('.safetensors') and remote['blob'] != row['sha256']:
        raise ValueError('published shard is not an LFS object: ' + name)
    return dict(bytes=remote['bytes'], blob=remote['blob'])


def committed_names(tree, files, *, fixed=None, ignore=()):
    """Names of ``files`` an interrupted publication already committed.

    Every other remote entry must be an exact ``fixed`` row (a server-side
    copy) or ignored metadata; anything foreign or different refuses resume.
    """
    fixed = fixed or {}
    done = set()
    for name, row in tree.items():
        if name in ignore:
            continue
        if name in fixed:
            if row != fixed[name]:
                raise ValueError('remote repository changed during upload: ' + name)
        elif name in files and matches(row, files[name]):
            done.add(name)
        else:
            raise ValueError('remote repository changed during upload: ' + name)
    return done


def receipt_operation(name, row):
    """Build an add whose size and SHA256 come from the sealed receipt."""
    from huggingface_hub import CommitOperationAdd

    operation = CommitOperationAdd(name, str(row['path']))
    info = operation.upload_info
    if info.size != row['bytes']:
        raise ValueError('HTTP LFS operation differs from sealed receipt: ' + name)
    digest = row.get('sha256')
    if digest is not None:
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid receipt SHA256: ' + name)
        digest = bytes.fromhex(digest)
        if info.is_hashed and info.sha256 != digest:
            raise ValueError('HTTP LFS operation differs from sealed receipt: ' + name)
        info.sha256 = digest  # Never mark uploaded/committed: the SDK owns those states.
    return operation


def folder_files(folder, ignore_patterns):
    """Regular files ``upload_large_folder`` would select, as {name: path}."""
    from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects

    folder = Path(folder)
    found = {}
    for directory, names, leaves in os.walk(folder, followlinks=False):
        for name in names:
            if (Path(directory) / name).is_symlink():
                raise ValueError('refuse symlink publication directory')
        for name in leaves:
            path = Path(directory) / name
            found[path.relative_to(folder).as_posix()] = path
    selected = set(filter_repo_objects(found, ignore_patterns=[*ignore_patterns, *DEFAULT_IGNORE_PATTERNS]))
    files = {}
    for name in sorted(selected):
        if found[name].is_symlink() or not found[name].is_file():
            raise ValueError('publication payload must contain only regular files: ' + name)
        files[name] = found[name]
    return files


def upload_files(api, repo_id, rows, *, commit_message, options=None):
    """Verified, resumable HTTP LFS upload of explicit rows; returns the head revision.

    ``rows`` maps names to dict(path, bytes, sha256[, git_blob, local]). The
    repository may hold only these files (plus the Hub's attributes); each
    committed file must match its row, and an interrupted run skips them.
    """
    require_qualified_hub()
    options = options or UploadOptions()
    rows = {name: dict(row, path=str(row['path']), local=row.get('local') or fingerprint(row['path']))
            for name, row in rows.items()}
    check_fingerprints(rows, rows)
    info = api.repo_info(repo_id, repo_type='model')
    if info.private:
        raise ValueError('published model repository is not public')
    tree = remote_tree(api, repo_id, info.sha)
    # The Hub rewrites attributes; ours ride the first batch with the largest shards.
    done = committed_names(tree, rows, ignore={'.gitattributes'})
    if '.gitattributes' in rows and (done or matches(tree.get('.gitattributes', {}), rows['.gitattributes'])):
        done.add('.gitattributes')

    def verify(commit, batch, operations):
        done.update(batch)
        remote = remote_tree(api, repo_id, commit)
        if set(remote) - {'.gitattributes'} != done - {'.gitattributes'}:
            raise ValueError('HTTP LFS batch inventory differs from expected files')
        for name in done - {'.gitattributes'}:
            resolved(remote[name], rows[name], name)

    pending = [name for name in upload_order(rows) if name not in done]
    if not pending:
        return info.sha
    return commit_batches(api, repo_id, rows, pending, revision=info.sha, commit_message=commit_message,
                          options=options, check_sources=lambda batch: check_fingerprints(rows, batch),
                          verify=verify)


def core_rows(files):
    """Adapt core publishers' {path, size, sha256, blob_id} rows to upload rows."""
    return {name: dict(path=str(row['path']), bytes=row['size'], sha256=row['sha256'], git_blob=row.get('blob_id'))
            for name, row in files.items()}


def upload_folder(api, repo_id, folder, *, ignore_patterns, state, commit_message, digests=None, options=None):
    """Verified, resumable HTTP LFS replacement for ``upload_large_folder``.

    ``digests`` maps names to writer SHA256s (weights, never reread); other
    files are hashed once into ``state`` and reused on resume.
    """
    options = options or UploadOptions()
    paths = folder_files(folder, ignore_patterns)
    digests = digests or {}
    rows = hashed_rows({n: p for n, p in paths.items() if n not in digests}, Path(state) / 'upload-hashes.json',
                       workers=options.workers)
    for name, digest in digests.items():
        if name not in paths:
            raise ValueError('receipt names a file outside the publication: ' + name)
        rows[name] = dict(path=str(paths[name]), bytes=paths[name].stat().st_size, sha256=digest)
    return upload_files(api, repo_id, rows, commit_message=commit_message, options=options)


def commit_batches(api, repo_id, files, names, *, revision, commit_message, check_sources, verify,
                   options, leading=(), log=None, operation=receipt_operation):
    """Commit ``names`` in order as parent-guarded batches; return the last revision.

    ``check_sources(batch)`` runs before and after each transfer and must raise
    if a source changed. ``verify(commit, batch, operations)`` must raise unless
    the new revision holds exactly what the publisher expects so far.
    ``leading`` operations (server-side copies) join the first commit.
    ``operation(name, row)`` builds each add (default: from the sealed receipt).
    """
    require_qualified_hub()
    log = log or (lambda message: print(message, flush=True))
    names, leading = list(names), list(leading)
    if len(names) != len(set(names)) or set(names) - set(files):
        raise ValueError('HTTP LFS batch names differ from publication inventory')
    total, done = sum(files[name]['bytes'] for name in names), 0
    for offset in range(0, max(len(names), 1 if leading else 0), options.batch_files):
        batch = names[offset:offset + options.batch_files]
        check_sources(batch)
        operations = [*leading, *(operation(name, files[name]) for name in batch)]
        leading = []
        size = sum(files[name]['bytes'] for name in batch)
        log(f'HTTP LFS batch: {len(batch)} files, {size} bytes, {options.workers} workers')
        started = time.monotonic()
        commit = api.create_commit(repo_id, operations, repo_type='model', revision='main',
                                   parent_commit=revision, num_threads=options.workers,
                                   commit_message=commit_message).oid
        elapsed = max(time.monotonic() - started, 1e-9)
        check_sources(batch)
        verify(commit, batch, operations)
        revision, done = commit, done + size
        log(f'HTTP LFS committed: {offset + len(batch)}/{len(names)} files, {done}/{total} bytes, '
            f'{elapsed:.0f} s, {size * 8 / elapsed / 1e6:.0f} Mbps, {revision}')
    return revision
