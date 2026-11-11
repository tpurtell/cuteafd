"""Explicit post-publication ownership transfer; never downloads checkpoint files.

Stage verified local files as hard links, import offline without copying, verify
all snapshot hashes and readiness, then separately authorize owned-tree cleanup.
Inherited files use supported metadata-only hard links entirely inside SparkNest;
parent and destination Hub OIDs and sizes must agree before linking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess

from kimi_k3.uniform_k2.storage import atomic_json, sha256_file


def load(path):
    return json.loads(Path(path).read_text())


def contained(path, root):
    return path == root or root in path.parents


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in path.parts):
        raise ValueError('invalid published relative path: ' + name)
    return path


def verified_inputs(packet_path, audit_path):
    packet, audit = load(packet_path), load(audit_path)
    if (audit['packet_sha256'] != sha256_file(packet_path)
            or audit['repo_id'] != packet['repo_id']
            or audit['status'] != 'anonymous-exact-tree-sizes-and-payload-identities-verified'):
        raise ValueError('publication audit does not match approved packet')
    files = packet['new_files'] | packet['copy_files']
    if files.keys() != audit['files'].keys():
        raise ValueError('published inventory differs')
    for name, row in files.items():
        relative(name)
        remote = audit['files'][name]
        if name != '.gitattributes' and (remote['bytes'], remote['sha256']) != (row['bytes'], row['sha256']):
            raise ValueError('published file differs: ' + name)
    commit = audit['commit']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('immutable published commit required')
    if len(packet['repo_id'].split('/')) != 2 or any(not p or p in ('.', '..') or '\\' in p for p in packet['repo_id'].split('/')):
        raise ValueError('invalid repo id')
    return packet, audit


def stage(package, destination, packet_path, audit_path, attributes, receipt_path, sources=None, inherited=None):
    """Preflight every file before creating any link; no EXDEV copy fallback."""
    package, destination = Path(package).resolve(), Path(destination).absolute()
    receipt_path = Path(receipt_path).absolute()
    packet, audit = verified_inputs(packet_path, audit_path)
    if contained(destination, package) or contained(package, destination) or contained(receipt_path, destination):
        raise ValueError('package, stage and receipts must be disjoint')
    if destination.exists():
        raise FileExistsError(destination)
    sources = sources or {}
    if not sources.keys() <= audit['files'].keys():
        raise ValueError('source override not published')
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    device = parent.stat().st_dev
    inherited = inherited or {}
    if not inherited.keys() <= packet['copies'].keys():
        raise ValueError('inherited alias must be an approved server copy')
    rows, blobs = {}, {}
    for name, remote in sorted(audit['files'].items()):
        if name in inherited:
            alias = inherited[name]
            if alias['bytes'] != remote['bytes'] or alias['sha256'] != remote['sha256']:
                raise ValueError('inherited alias proof differs: ' + name)
            rows[name] = dict(alias, inherited=True)
            continue
        source = Path(attributes if name == '.gitattributes' else sources.get(name, package / name)).resolve(strict=True)
        md = source.stat()
        if not stat.S_ISREG(md.st_mode) or md.st_dev != device:
            raise ValueError('source not on staging filesystem; no copying: ' + str(source))
        if md.st_size != remote['bytes'] or sha256_file(source) != remote['sha256']:
            raise ValueError('source differs from published file: ' + name)
        if name == '.gitattributes' or remote['storage'] != 'lfs':
            h = hashlib.sha1(f'blob {md.st_size}\0'.encode())
            with source.open('rb') as stream:
                while chunk := stream.read(4 << 20):
                    h.update(chunk)
            oid = h.hexdigest()
            if remote.get('git_blob', oid) != oid:
                raise ValueError('published git OID differs: ' + name)
        else:
            oid = remote['sha256']
        rows[name] = dict(source=str(source), oid=oid, bytes=md.st_size, sha256=remote['sha256'],
                          device=md.st_dev, inode=md.st_ino, mtime_ns=md.st_mtime_ns)
        blobs.setdefault(oid, source)
    repo = destination / ('models--' + packet['repo_id'].replace('/', '--'))
    (repo / 'blobs').mkdir(parents=True)
    snapshot = repo / 'snapshots' / audit['commit']
    snapshot.mkdir(parents=True)
    for oid, source in blobs.items():
        os.link(source, repo / 'blobs' / oid)
    for name, row in rows.items():
        if row.get('inherited'):
            continue
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(repo / 'blobs' / row['oid'], path.parent))
        md = path.stat()
        if (md.st_dev, md.st_ino, md.st_mtime_ns) != (row['device'], row['inode'], row['mtime_ns']):
            raise ValueError('source changed during staging: ' + name)
    (repo / 'refs').mkdir()
    (repo / 'refs/main').write_text(audit['commit'])
    receipt = dict(schema='cuteafd-postpublish-stage-v1', status='hardlinks-staged',
        repo_id=packet['repo_id'], commit=audit['commit'], repo=str(repo), package=str(package),
        packet_sha256=sha256_file(packet_path), audit_sha256=sha256_file(audit_path),
        files=rows, copied_bytes=0, fetched_bytes=0)
    atomic_json(receipt_path, receipt)
    return receipt


def inherited_proofs(packet_path, audit_path, hub):
    """Prove server-copy identities from both pinned Hub listings, no payload IO."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    packet, audit = verified_inputs(packet_path, audit_path)
    api = HfApi(token=False)
    trees = {}
    for repo, revision in ((packet['parent_repo'], packet['parent_revision']), (packet['repo_id'], audit['commit'])):
        trees[repo] = {v.path: v for v in api.list_repo_tree(repo, revision=revision, recursive=True) if isinstance(v, RepoFile)}
    parent = Path(hub).resolve() / ('models--' + packet['parent_repo'].replace('/', '--')) / 'snapshots' / packet['parent_revision']
    proofs = {}
    for name, source in packet['copies'].items():
        relative(source)
        a, b = trees[packet['parent_repo']][source], trees[packet['repo_id']][name]
        oid = a.lfs.sha256 if a.lfs else a.blob_id
        other = b.lfs.sha256 if b.lfs else b.blob_id
        expected = audit['files'][name]
        if a.size != b.size or a.size != expected['bytes'] or oid != other:
            raise ValueError('pinned parent/destination OIDs differ: ' + name)
        if (a.lfs and oid != expected['sha256']) or (not a.lfs and oid != expected['git_blob']):
            raise ValueError('Hub identity differs from anonymous audit: ' + name)
        path = parent / source
        if path.stat().st_size != a.size:
            raise ValueError('parent snapshot size differs: ' + name)
        proofs[name] = dict(source=str(path), oid=oid, bytes=a.size, sha256=expected['sha256'],
            parent_repo=packet['parent_repo'], parent_revision=packet['parent_revision'])
    return proofs


def alias_inherited(stage_receipt, hub, receipt_path, *, limit=None):
    """Supported link(2) within the mount: same inode, replicas and payload."""
    staged = load(stage_receipt)
    hub = Path(hub).resolve()
    repo = hub / ('models--' + staged['repo_id'].replace('/', '--'))
    snapshot = repo / 'snapshots' / staged['commit']
    rows = [(n, r) for n, r in staged['files'].items() if r.get('inherited')]
    if limit is not None:
        rows = rows[:limit]
    done = {}
    (repo / 'blobs').mkdir(parents=True, exist_ok=True)
    for name, row in rows:
        source = Path(row['source'])
        if not contained(source.resolve(), hub):
            raise ValueError('alias source outside SparkNest hub')
        blob = repo / 'blobs' / row['oid']
        before = source.stat()
        if before.st_size != row['bytes']:
            raise ValueError('alias source size differs')
        if blob.exists():
            md = blob.stat()
            if (md.st_dev, md.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError('existing blob is not metadata alias: ' + name)
        else:
            os.link(source, blob, follow_symlinks=True)
        md = blob.stat()
        if (md.st_dev, md.st_ino) != (before.st_dev, before.st_ino) or md.st_nlink < 2:
            raise ValueError('metadata hardlink identity not preserved')
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        link = os.path.relpath(blob, target.parent)
        if target.is_symlink():
            if os.readlink(target) != link:
                raise ValueError('existing snapshot alias differs')
        elif target.exists():
            raise FileExistsError(target)
        else:
            target.symlink_to(link)
        done[name] = dict(inode=md.st_ino, nlink=md.st_nlink, device=md.st_dev, source=row['source'], blob=str(blob))
        atomic_json(receipt_path, dict(status='metadata-aliases-created', files=done, copied_bytes=0, fetched_bytes=0))
    return dict(status='metadata-aliases-created', files=done, copied_bytes=0, fetched_bytes=0)


def import_staged(stage_receipt, receipt_path, *, nest='nest'):
    staged = load(stage_receipt)
    if staged['status'] != 'hardlinks-staged':
        raise ValueError('completed staging receipt required')
    if Path(receipt_path).exists():
        raise FileExistsError(receipt_path)
    for name, row in staged['files'].items():
        if row.get('inherited'):
            continue
        path = Path(staged['repo']) / 'snapshots' / staged['commit'] / name
        md = path.stat()
        if (md.st_dev, md.st_ino, md.st_size, md.st_mtime_ns) != (row['device'], row['inode'], row['bytes'], row['mtime_ns']):
            raise ValueError('staged file changed: ' + name)
    command = [nest, 'hf', 'import', staged['repo'], '--offline', '--no-copy', '--json']
    start = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    job = json.loads(start.stdout)['job']
    # Keep one bounded blocking CLI wait, not an agent-side polling loop.
    result = subprocess.run([nest, 'jobs', str(job), '--wait', '--json'], capture_output=True,
                            text=True, check=True, timeout=7200)
    raw = json.loads(result.stdout)
    progress, terminal = raw['import'], raw['job']
    receipt = dict(schema='cuteafd-postpublish-import-v1', command=command, job=job,
        stage_sha256=sha256_file(stage_receipt), result=raw, copied_bytes=progress['copied_bytes'], fetched_bytes=0)
    atomic_json(receipt_path, receipt)
    if (terminal.get('error') or terminal.get('cancelled') or not terminal['finished']
            or progress['errors'] or progress.get('cancelled') or progress['copied_bytes'] != 0):
        raise ValueError('offline no-copy import did not complete cleanly; inspect receipt')
    receipt['status'] = 'offline-no-copy-imported'
    atomic_json(receipt_path, receipt)
    return receipt


def verify(stage_receipt, import_receipt, hub, planner, programs, v4, receipt_path):
    staged, imported = load(stage_receipt), load(import_receipt)
    if imported.get('status') != 'offline-no-copy-imported' or imported['stage_sha256'] != sha256_file(stage_receipt):
        raise ValueError('matching completed no-copy import required')
    hub = Path(hub).resolve()
    repo = hub / ('models--' + staged['repo_id'].replace('/', '--'))
    snapshot = repo / 'snapshots' / staged['commit']
    actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
    if actual != staged['files'].keys():
        raise ValueError('imported snapshot file inventory differs')
    for name, row in staged['files'].items():
        path = snapshot / name
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError('imported snapshot differs: ' + name)
    views = ['original', 'k8-compact', 'k6-compact'] if (snapshot / 'cuteafd-mc.json').exists() else [None]
    env = dict(os.environ, HF_HOME=str(hub.parent), CUTEAFD_DEPLOYMENT_TEST_PROGRAMS=str(programs),
               CUTEAFD_V4_PLACEMENT_INPUTS=str(v4))
    plans = {}
    for view in views:
        command = [str(planner), 'plan', staged['repo_id'], '--revision', staged['commit'],
            '--hf-home', str(hub.parent), '--layout', '--rtx', '2', '--spark-ranks', '6',
            '--workspace-manifest', str(programs), '--json', '--require-ready']
        if view:
            command += ['--core-view', view]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, env=env)
        plan = dict(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        plans[view or 'native'] = plan
        atomic_json(receipt_path, dict(status='verification-running', plans=plans))
        if result.returncode:
            raise ValueError('normal HF_HOME readiness failed; inspect receipt')
        parsed = json.loads(result.stdout)
        if not parsed['fits'] or parsed['config_error'] or parsed['missing_shards'] or parsed['numerics_deviations']:
            raise ValueError('imported checkpoint not ready')
    receipt = dict(schema='cuteafd-postpublish-verify-v1', status='imported-hashes-and-readiness-verified',
        stage_sha256=sha256_file(stage_receipt), import_sha256=sha256_file(import_receipt),
        repo_id=staged['repo_id'], commit=staged['commit'], snapshot=str(snapshot),
        verified_files=len(actual), plans=plans, copied_bytes=0, fetched_bytes=0)
    atomic_json(receipt_path, receipt)
    return receipt


def cleanup(verify_receipt, roots, receipt_path, protected):
    """Delete only explicit owned build roots, after verification and FD checks.

    Caller must supply every running job's source/dependency roots as protected.
    Logs and JSON evidence are preserved outside the removed trees before unlink.
    """
    verified = load(verify_receipt)
    if verified['status'] != 'imported-hashes-and-readiness-verified':
        raise ValueError('verified imported snapshot required before cleanup')
    build_root = (Path.home() / '.cache/cuteafd/builds').resolve()
    roots = [Path(p).absolute() for p in roots]
    receipt_path = Path(receipt_path).absolute()
    protected = [Path(p).resolve() for p in protected] + [Path(verified['snapshot']).resolve()]
    for root in roots:
        if (root.is_symlink() or root.resolve() != root or root == build_root or not contained(root, build_root)
                or contained(receipt_path, root) or contained(Path(verify_receipt).resolve(), root)
                or any(contained(p, root) or contained(root, p) for p in protected)):
            raise ValueError('cleanup root is unsafe or protected: ' + str(root))
    if any(a != b and contained(a, b) for a in roots for b in roots):
        raise ValueError('overlapping cleanup roots')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            entries = [proc / 'cwd', proc / 'exe'] + list((proc / 'fd').iterdir())
            targets = [p.resolve() for p in entries]
            targets += [Path(line.split(None, 5)[5].strip()) for line in (proc / 'maps').read_text().splitlines()
                        if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')]
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        if any(contained(p, root) for p in targets for root in roots):
            raise ValueError('process ' + proc.name + ' still references cleanup root')
    evidence = receipt_path.parent / (receipt_path.stem + '-retained')
    evidence.mkdir(parents=True, exist_ok=False)
    before = shutil.disk_usage(build_root).free
    logical = 0
    inodes = {}
    for i, root in enumerate(roots):
        for path in root.rglob('*'):
            if path.is_symlink() or not path.is_file():
                continue
            md = path.stat()
            logical += md.st_size
            key = (md.st_dev, md.st_ino)
            entry = inodes.setdefault(key, dict(bytes=md.st_size, nlink=md.st_nlink, removed_links=0))
            entry['removed_links'] += 1
            if path.suffix in ('.json', '.jsonl', '.log', '.stderr', '.txt', '.py', '.sh'):
                dest = evidence / str(i) / path.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    intent = dict(schema='cuteafd-postpublish-cleanup-v1', status='cleanup-started',
        verification_sha256=sha256_file(verify_receipt), roots=list(map(str, roots)),
        protected=list(map(str, protected)), retained=str(evidence), logical_unlinked_bytes=logical,
        retained_hardlink_payload_bytes=sum(r['bytes'] for r in inodes.values() if r['nlink'] > r['removed_links']),
        removed_last_link_intermediate_bytes=sum(r['bytes'] for r in inodes.values() if r['nlink'] == r['removed_links']),
        byte_accounting_note='Unique inode sizes; external hardlinks retain payload, last-link bytes are not a physical reclaim measurement.',
        free_before=before)
    atomic_json(receipt_path, intent)
    for root in roots:
        shutil.rmtree(root)
    intent.update(status='owned-intermediates-removed', free_after=shutil.disk_usage(build_root).free)
    intent['observed_free_delta_bytes'] = intent['free_after'] - before
    intent['free_delta_note'] = 'Host-wide free-space delta, affected by concurrent work; hardlinks are not reclaimed payload.'
    atomic_json(receipt_path, intent)
    return intent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    s = commands.add_parser('stage')
    for key in ('package', 'destination', 'packet', 'audit', 'attributes', 'receipt'):
        s.add_argument('--' + key, type=Path, required=True)
    s.add_argument('--sources', type=Path)
    s.add_argument('--inherited', type=Path)
    a = commands.add_parser('alias')
    for key in ('stage-receipt', 'hub', 'receipt'):
        a.add_argument('--' + key, type=Path, required=True)
    a.add_argument('--limit', type=int)
    p = commands.add_parser('prove-inherited')
    for key in ('packet', 'audit', 'hub', 'receipt'):
        p.add_argument('--' + key, type=Path, required=True)
    i = commands.add_parser('import')
    i.add_argument('--stage-receipt', type=Path, required=True)
    i.add_argument('--receipt', type=Path, required=True)
    i.add_argument('--nest', default='nest', help='Installed nest executable (absolute path for service environments)')
    v = commands.add_parser('verify')
    for key in ('stage-receipt', 'import-receipt', 'hub', 'planner', 'programs', 'v4', 'receipt'):
        v.add_argument('--' + key, type=Path, required=True)
    c = commands.add_parser('cleanup')
    for key in ('verify-receipt', 'receipt'):
        c.add_argument('--' + key, type=Path, required=True)
    c.add_argument('--root', type=Path, action='append', required=True)
    c.add_argument('--protected', type=Path, action='append', required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    args['receipt_path'] = args.pop('receipt')
    if action in ('stage', 'prove-inherited'):
        args['packet_path'], args['audit_path'] = args.pop('packet'), args.pop('audit')
    if action == 'prove-inherited':
        receipt = args.pop('receipt_path')
        result = inherited_proofs(**args)
        atomic_json(receipt, result)
        print(json.dumps(dict(status='pinned-Hub-copy-identities-proven', files=len(result))))
        return
    if action == 'stage':
        args['sources'] = load(args['sources']) if args['sources'] else None
        args['inherited'] = load(args['inherited']) if args['inherited'] else None
    if action == 'cleanup':
        args['roots'] = args.pop('root')
    result = {'stage': stage, 'alias': alias_inherited, 'import': import_staged, 'verify': verify, 'cleanup': cleanup}[action](**args)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'plans', 'result')}, sort_keys=True))


if __name__ == '__main__':
    main()
