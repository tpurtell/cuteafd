"""Offline no-copy adoption, native eight-place spread and task-scoped retirement."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.machinery
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'inkling2'

from kimi_k3.uniform_k2.storage import atomic_json, file_identity, fsync_directory
from .publish import TARGET_REPO, load, manifest_digest, regular, relative

TARGET_ARGS = ('raptor:d2', 'raptor:d3', 'ostrich', 'dodo', 'emu', 'kiwi', 'rhea', 'moa')
TARGETS = ('raptor (D2)', 'raptor (D3)', *TARGET_ARGS[2:])
NAMESPACE = Path('/hf-home/hub')
MANAGED = Path('/mnt/sparknest/hf-home/hub')
MODEL = 'models--' + TARGET_REPO.replace('/', '--')


def native(*args, timeout=21600):
    result = subprocess.run(['nest', '--json', *map(str, args)], check=True,
                            capture_output=True, text=True, timeout=timeout)
    raw = result.stdout
    notice = TARGET_REPO + ': mirrored\n'
    if args[:2] == ('hf', 'import') and raw.startswith(notice):
        raw = raw[len(notice):]
    return json.loads(raw)


def successful(result, *, importing_files=None, eviction_files=None):
    """Accept the known pure-eviction host flag quirk only with exact counts."""
    job = result.get('job', {})
    if (job.get('finished') is not True or job.get('error') or job.get('cancelled')
            or job.get('pending', 0) or any(': failed' in n for n in job.get('notes', []))):
        raise ValueError('native SparkNest job failed or is incomplete')
    hosts = job.get('hosts', {})
    if not hosts:
        raise ValueError('native job has no completed host counters')
    if any((eviction_files is None and h.get('finished') is not True) or h.get('failed')
           or h.get('cancelled') or h.get('done_files', 0) != h.get('total_files', 0)
           or (h.get('total_bytes', 0) > 0 and h.get('done_bytes', 0) != h['total_bytes'])
           for h in hosts.values()):
        raise ValueError('native SparkNest host job failed or is incomplete')
    if eviction_files is not None and sum(h.get('done_files', 0) for h in hosts.values()) != eviction_files:
        raise ValueError('native pure-eviction count differs from plan')
    if importing_files is not None:
        stat = result.get('import', {})
        counts = [stat.get(k, 0) for k in ('files', 'adopted', 'skipped')]
        if (stat.get('finished') is not True or stat.get('errors') != [] or stat.get('cancelled')
                or stat.get('copied_bytes') != 0 or any(type(c) is not int or c < 0 for c in counts)
                or sum(counts) != importing_files):
            raise ValueError('offline no-copy import was not verified')
    return result


def check_plan(plan, revision, files, *, offgrid=False):
    goal = plan.get('goal', {})
    if (plan.get('feasible') is not True or plan.get('blocked')
            or goal.get('goal') != ('offgrid' if offgrid else 'place')
            or goal.get('selector') != 'hf:' + TARGET_REPO or goal.get('hub') != str(NAMESPACE)):
        raise ValueError('placement is not restricted to this checkpoint')
    if offgrid:
        if goal.get('hosts') != []:
            raise ValueError('off-grid retirement must not add placement hosts')
    elif (len(goal.get('hosts', [])) != 8 or set(goal['hosts']) != set(TARGETS)
            or goal.get('factor') != 1 / 8 or goal.get('by_speed', False)):
        raise ValueError('placement must use the authorized even eight-place spread')
    snapshot = NAMESPACE / MODEL / 'snapshots' / revision
    refs = NAMESPACE / MODEL / 'refs'
    for step in plan.get('steps', []):
        if step.get('kind') not in (('evict',) if offgrid else ('replicate', 'evict')):
            raise ValueError('unexpected native placement operation')
        if step.get('kind') == 'replicate' and step.get('host') not in TARGETS:
            raise ValueError('placement writes to unauthorized disk')
        if offgrid and step.get('host') in TARGETS:
            raise ValueError('off-grid retirement cannot evict an authorized fabric copy')
        for copy in step.get('copies', []):
            path = copy.get('path')
            if path == str(refs / 'main'):
                valid = copy.get('group') == str(refs) and copy.get('size') == len(revision)
            else:
                name = copy.get('name', '')
                relative(name)
                valid = (name in files and path == str(snapshot / name)
                    and copy.get('group') == TARGET_REPO and copy.get('size') == files[name]['bytes'])
            if not valid or copy.get('skip'):
                raise ValueError('placement touches an unreviewed file or revision')
    return plan


def stage_snapshot(published, stage_root):
    """Create a conventional HF snapshot using hardlinks only; EXDEV is fatal."""
    if (published.get('status') != 'passed' or published.get('repo_id') != TARGET_REPO
            or not re.fullmatch('[0-9a-f]{40,64}', published.get('revision', ''))
            or manifest_digest(published['files']) != published.get('manifest_sha256')):
        raise ValueError('matching verified publication required')
    stage_root = Path(stage_root).absolute()
    for parent in (stage_root, *stage_root.parents):
        if parent.is_symlink():
            raise ValueError('stage root cannot traverse a symlink')
    if stage_root == Path(published['output']) or stage_root in Path(published['output']).parents or Path(published['output']) in stage_root.parents:
        raise ValueError('standalone stage and export must be disjoint')
    stamp = dict(repo_id=TARGET_REPO, revision=published['revision'], manifest_sha256=published['manifest_sha256'])
    owner = stage_root / '.inkling2-owned-stage.json'
    if owner.exists():
        regular(owner)
        if load(owner) != stamp:
            raise ValueError('stage ownership differs from publication')
    elif stage_root.exists() and any(stage_root.iterdir()):
        raise ValueError('new stage root must be an empty task-owned directory')
    stage_root.mkdir(parents=True, exist_ok=True)
    device = stage_root.stat().st_dev
    # Preflight every source before creating any model links.
    for name, row in published['files'].items():
        relative(name)
        source = regular(row['path'])
        if (file_identity(source) != row['file_identity'] or source.stat().st_dev != row['device']
                or source.stat().st_dev != device):
            raise ValueError('staging source changed or crosses filesystems; no copy fallback')
    atomic_json(owner, stamp)
    repo = stage_root / MODEL
    snapshot = repo / 'snapshots' / published['revision']
    if any(p.is_symlink() for p in (snapshot, *snapshot.parents)):
        raise ValueError('stage directories cannot traverse symlinks')
    if repo.exists() and {p.name for p in repo.iterdir()} - {'snapshots', 'refs'}:
        raise ValueError('unexpected files in owned HF staging repository')
    if (repo / 'snapshots').exists() and {p.name for p in (repo / 'snapshots').iterdir()} - {published['revision']}:
        raise ValueError('unexpected revision in owned HF staging repository')
    snapshot.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in snapshot.rglob('*')) or {
            p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()} - set(published['files']):
        raise ValueError('unexpected files in staged snapshot')
    for name, row in published['files'].items():
        source, target = Path(row['path']), snapshot / name
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError('stage file cannot traverse symlinks')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError('existing stage file is not the published hardlink')
        else:
            os.link(source, target)  # Never copy/reflink/download behind the caller's back.
        if file_identity(target) != row['file_identity']:
            raise ValueError('stage hardlink identity changed')
    refs = repo / 'refs'
    if refs.is_symlink() or (refs.exists() and {p.name for p in refs.iterdir()} - {'main'}):
        raise ValueError('unexpected staged HF refs')
    refs.mkdir(exist_ok=True)
    main = refs / 'main'
    if main.is_symlink() or (main.exists() and main.read_text() != published['revision']):
        raise ValueError('existing stage main ref differs')
    with main.open('w') as stream:
        stream.write(published['revision']); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(refs)
    return repo


def verify_managed(files, revision, *, call=native, managed=MANAGED, allow_offgrid=False):
    """Inspect only native placement metadata and the tiny main ref."""
    snapshot = NAMESPACE / MODEL / 'snapshots' / revision
    seen, distribution = set(), dict.fromkeys(TARGETS, 0)
    largest = max((r['bytes'] for n, r in files.items() if n.endswith('.safetensors')), default=0)
    for directory in sorted({str(Path(n).parent) for n in files}):
        for row in call('ls', snapshot / directory)['entries']:
            if row.get('kind') == 'directory':
                continue
            name = (Path(directory) / row['name']).as_posix()
            hosts = row.get('hosts', [])
            placed = set(hosts) & set(TARGETS)
            if (name not in files or row.get('kind') != 'regular' or row.get('size') != files[name]['bytes']
                    or row.get('sealed') is not True or row.get('writing_on') is not None
                    or len(placed) != 1 or (not allow_offgrid and len(hosts) != 1) or name in seen):
                raise ValueError('managed one-copy file inventory differs: ' + name)
            seen.add(name)
            if name.endswith('.safetensors'):
                distribution[next(iter(placed))] += row['size']
    if (seen != set(files) or not all(distribution.values())
            or max(distribution.values()) - min(distribution.values()) > largest):
        raise ValueError('managed shard spread does not cover all eight targets evenly')
    if (Path(managed) / MODEL / 'refs/main').read_text().strip() != revision:
        raise ValueError('managed main ref revision differs')
    refs = call('ls', NAMESPACE / MODEL / 'refs')['entries']
    if (len(refs) != 1 or refs[0].get('name') != 'main' or refs[0].get('kind') != 'regular'
            or refs[0].get('size') != len(revision) or refs[0].get('writing_on') is not None
            or len(set(refs[0].get('hosts', [])) & set(TARGETS)) != 1
            or (not allow_offgrid and len(refs[0]['hosts']) != 1)):
        raise ValueError('managed main ref placement differs')
    return distribution


def handoff(root, stage_root, *, call=native, managed=MANAGED):
    root, stage_root = Path(root).resolve(), Path(stage_root).absolute()
    published = load(root / 'publication-complete.json')
    identity = {k: published[k] for k in ('repo_id', 'revision', 'manifest_sha256')}
    if (published.get('status') != 'passed' or published['repo_id'] != TARGET_REPO
            or not re.fullmatch('[0-9a-f]{40,64}', published.get('revision', ''))
            or manifest_digest(published['files']) != published['manifest_sha256']):
        raise ValueError('verified publication required before handoff')
    with (root / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / 'handoff-complete.json'
        if marker.exists():
            record = load(marker)
            if record.get('identity') != identity or record.get('status') != 'passed':
                raise ValueError('handoff completion identity differs')
            return record
        state_path = root / 'handoff-state.json'
        state = load(state_path) if state_path.exists() else dict(identity=identity, stage_root=str(stage_root))
        if state.get('identity') != identity or state.get('stage_root') != str(stage_root):
            raise ValueError('handoff recovery identity differs')
        atomic_json(state_path, state)

        def job(key, command, **checks):
            complete = root / (key + '-complete.json')
            if complete.exists():
                return successful(load(complete), **checks)
            if key not in state:
                number = call(*command).get('job')
                if type(number) is not int or number <= 0:
                    raise ValueError('native job id missing')
                state[key] = number
                atomic_json(state_path, state)
            result = successful(call('jobs', state[key], '--wait'), **checks)
            atomic_json(complete, result)
            return result

        if 'import-job' not in state:
            repo = stage_snapshot(published, stage_root)
        else:
            repo = stage_root / MODEL
        job('import-job', ('hf', 'import', repo, '--offline', '--no-copy'), importing_files=len(published['files']))

        def placement(key, arguments, *, offgrid=False):
            path = root / (key + '.json')
            if path.exists():
                plan = check_plan(load(path), published['revision'], published['files'], offgrid=offgrid)
            else:
                plan = check_plan(call('plan', *arguments), published['revision'], published['files'], offgrid=offgrid)
                atomic_json(path, plan)
            if plan.get('steps') and not (root / (key + '-job-complete.json')).exists():
                # Inspect exactly the plan that will be applied, including all removals.
                shown = check_plan(call('plan', 'show', plan['id']), published['revision'], published['files'], offgrid=offgrid)
                if shown != plan:
                    raise ValueError('native plan changed before apply')
                eviction = sum(len(s['copies']) for s in plan['steps']) if all(s['kind'] == 'evict' for s in plan['steps']) else None
                job(key + '-job', ('plan', 'apply', plan['id']), eviction_files=eviction)
            return plan

        placement('placement', ('--place', 'hf:' + TARGET_REPO, '--factor', '1/8',
                               *(v for host in TARGET_ARGS for v in ('--host', host))))
        verify_managed(published['files'], published['revision'], call=call, managed=managed, allow_offgrid=True)
        placement('offgrid', ('--offgrid', 'hf:' + TARGET_REPO), offgrid=True)
        distribution = verify_managed(published['files'], published['revision'], call=call, managed=managed)
        record = dict(schema='cuteafd-inkling2-handoff-complete-v1', status='passed', identity=identity,
            repo_id=TARGET_REPO, revision=published['revision'], files=len(published['files']), copied_bytes=0,
            stage_root=str(stage_root), targets=list(TARGETS), distribution_bytes=distribution,
            managed_snapshot=str(Path(managed) / MODEL / 'snapshots' / published['revision']),
            one_copy_per_file=True, time=datetime.now(timezone.utc).isoformat())
        atomic_json(marker, record)
        return record


def assert_unused(paths):
    """Reuse the repository's process/container scanner, failing closed on gaps."""
    script = Path(__file__).resolve().parents[2] / 'scripts/dev/cache-gc'
    loader = importlib.machinery.SourceFileLoader('_inkling2_cache_gc', str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    gc = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = gc
    loader.exec_module(gc)
    mounts = gc.docker_mounts()
    busy, partial = gc.process_paths(Path('/'))
    if mounts is None:
        raise RuntimeError('cannot inspect containers before retirement')
    if partial:
        # WSL login infrastructure and the capability-bearing SparkNest daemon
        # can be non-dumpable to their own uid. Match only the known programs;
        # terminal native import/placement receipts attest daemon completion.
        for proc in Path('/proc').iterdir():
            try:
                if not proc.name.isdigit() or proc.stat().st_uid != os.getuid():
                    continue
                unreadable = False
                for name in ('cwd', 'root', 'exe'):
                    try:
                        os.readlink(proc / name)
                    except PermissionError:
                        unreadable = True
                try:
                    with os.scandir(proc / 'fd') as descriptors:
                        list(descriptors)
                except PermissionError:
                    unreadable = True
                for name in ('maps', 'environ'):
                    try:
                        (proc / name).read_bytes()
                    except PermissionError:
                        unreadable = True
                if not unreadable:
                    continue
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
                known = (argv[:2] == [b'/usr/lib/systemd/systemd', b'--user']
                    or argv[0] == b'(sd-pam)'
                    or argv[0].strip() == b'sshd-session: tj@notty'
                    or argv[:3] == [b'/home/linuxbrew/.linuxbrew/opt/sparknest/bin/sparknestd',
                                    b'--config', b'/srv/sparknest/node.toml'])
                if not known:
                    raise RuntimeError('cannot inspect user process before cleanup: ' + proc.name)
            except FileNotFoundError:
                continue
    for path in paths:
        if any(gc.inside(p, path) for p in busy) or any(gc.related(p, path) for p in mounts):
            raise RuntimeError('task leaf is in use by a process or container: ' + str(path))


def retire(root, *, intermediates=(), build_root=None, inspect=assert_unused):
    """Delete only explicit task payload leaves after terminal native placement."""
    root = Path(root).resolve()
    published, complete = (load(root / n) for n in ('publication-complete.json', 'handoff-complete.json'))
    identity = {k: published[k] for k in ('repo_id', 'revision', 'manifest_sha256')}
    if (published.get('status') != 'passed' or complete.get('status') != 'passed'
            or complete.get('identity') != identity or complete.get('one_copy_per_file') is not True):
        raise ValueError('verified matching handoff is required for retirement')
    output, stage = Path(published['output']), Path(complete['stage_root'])
    if stage.exists() and load(stage / '.inkling2-owned-stage.json') != identity:
        raise ValueError('stage ownership differs before retirement')
    if output.exists() and load(output / 'package-state.json').get('identity') != published['identity']:
        raise ValueError('standalone output ownership differs before retirement')
    paths = [output, stage]
    allowed = {'frontier', 'stages', 'projections', 'expert-projections', 'banks', 'candidates',
               'packed', 'activations', 'target-taps', 'routes'}
    for value in intermediates:
        p = Path(value).absolute()
        if root not in p.parents or p.name not in allowed:
            raise ValueError('intermediate must be an explicit task-owned payload leaf')
        paths.append(p)
    if build_root is not None:
        build = Path(build_root).absolute()
        if build.parent != Path.home() / '.cache/cuteafd/builds' or not build.name.startswith('inkling2-'):
            raise ValueError('build cleanup requires this task build root')
        paths.extend(p for p in build.iterdir() if p.name.startswith('target') or p.name in
                     ('exllamav3', 'tmp', 'release-staging', 'torch-extensions', 'cuda-cache'))
    protected = [Path(published['source_snapshot']).resolve(), MANAGED.resolve(), root]
    present = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        resolved = path.resolve()
        if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
                or not path.is_dir() or path.stat().st_uid != os.getuid()
                or any(resolved == p or resolved in p.parents or (p != root and p in resolved.parents) for p in protected)):
            raise ValueError('cleanup path is protected or not an owned standalone directory')
        present.append(resolved)
    inspect(present)
    for path in sorted(set(present), key=lambda p: len(p.parts), reverse=True):
        shutil.rmtree(path)
    record = dict(status='passed', identity=identity, paths=list(map(str, present)),
                  source_and_shared_stores_preserved=True, time=datetime.now(timezone.utc).isoformat())
    atomic_json(root / 'handoff-cleanup-complete.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage-root', type=Path)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--cleanup-request', type=Path, help='JSON containing explicit intermediates and optional build_root')
    args = parser.parse_args()
    if args.cleanup:
        request = load(args.cleanup_request) if args.cleanup_request else {}
        result = retire(args.root, **request)
    elif args.stage_root:
        result = handoff(args.root, args.stage_root)
    else:
        parser.error('handoff requires --stage-root; cleanup requires --cleanup')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
