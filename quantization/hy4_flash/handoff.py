"""Adopt the published Hy4-flash snapshot and verify native SparkNest placement."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import re

from inkling2 import handoff as nest
from inkling2 import publish as hub
from kimi_k3.uniform_k2.storage import atomic_json, file_identity

from .publish import TARGET, TARGET_REPO, pinned_files

MODEL = 'models--' + TARGET_REPO.replace('/', '--')
PIN_HOST = 'raptor'
ADOPTION_HOST = 'maxq'
TARGETS = frozenset(nest.TARGETS)
KEEP_REF = 'keep/hy4-k4-mc'


@dataclass(frozen=True)
class Placement:
    target: hub.PublicationTarget
    build_prefix: str
    engram: bool = False
    adoption_host: str = ADOPTION_HOST
    keep_ref: str = KEEP_REF

    @property
    def model(self):
        return 'models--' + self.target.repo_id.replace('/', '--')

    @property
    def schema(self):
        return f'cuteafd-{self.target.family}-handoff-complete-v2'


FLASH = Placement(TARGET, 'hy4-flash-k4-mc', engram=True)


def _publication(root, output, layout=FLASH):
    record = hub.load(root / 'publication-complete.json')
    identity = {k: record[k] for k in ('repo_id', 'revision', 'manifest_sha256')}
    if (record.get('schema') != layout.target.schema('complete') or record.get('status') != 'passed'
            or identity['repo_id'] != layout.target.repo_id or record.get('output') != str(output)
            or not re.fullmatch('[0-9a-f]{40,64}', identity['revision'])
            or hub.manifest_digest(record['files']) != identity['manifest_sha256']):
        raise ValueError('matching verified Hy4 publication required before handoff')
    return record, identity


def _inventory(call, managed, files, revision, *, physical_paths=None, model=MODEL, keep_ref=KEEP_REF):
    """Use native namespace metadata and read only the two small revision refs."""
    repo = nest.NAMESPACE / model
    snapshot = repo / 'snapshots' / revision
    snapshots = call('ls', snapshot.parent)['entries']
    if [(row['name'], row['kind']) for row in snapshots] != [(revision, 'directory')]:
        raise ValueError('managed checkpoint must contain only the published snapshot')
    blobs = {row['name']: row for row in call('ls', repo / 'blobs')['entries']}
    if set(blobs) != {row['hub_oid'] for row in files.values()}:
        raise ValueError('managed blob inventory differs from publication')
    physical_by_oid = {}
    for oid, meta in blobs.items():
        physical = repo / 'blobs' / oid
        if meta.get('kind') == 'symlink':
            # SparkNest's no-copy import mirrors the HF alias and may migrate
            # its physical file to the deduplicated hub-wide store.
            target = meta.get('target', '')
            match = re.fullmatch(r'\.\./\.\./blobs/([a-f0-9]{2})/([a-f0-9]{64})', target)
            if not match or not match[2].startswith(match[1]):
                raise ValueError('invalid managed shared blob alias: ' + oid)
            physical = nest.NAMESPACE / 'blobs' / match[1] / match[2]
            rows = call('ls', physical)['entries']
            if len(rows) != 1 or rows[0]['name'] != physical.name:
                raise ValueError('managed shared blob does not resolve: ' + oid)
            meta = rows[0]
        physical_by_oid[oid] = (physical, meta)
    result = {}
    directories = {parent for name in files for parent in Path(name).parents}
    for directory in sorted(directories):
        for row in call('ls', snapshot / directory)['entries']:
            path = directory / row['name']
            name = path.as_posix()
            if row.get('kind') == 'directory' and path in directories:
                continue
            if name not in files or name in result or row.get('kind') != 'symlink':
                raise ValueError('managed snapshot file inventory differs: ' + name)
            oid = files[name]['hub_oid']
            physical, blob = physical_by_oid[oid]
            expected = os.path.relpath(repo / 'blobs' / oid, snapshot / directory)
            if (row.get('target') != expected or blob.get('kind') != 'regular' or not blob['hosts']
                    or blob.get('sealed') is not True or blob.get('writing_on') is not None
                    or blob['size'] != files[name]['bytes']):
                raise ValueError('managed snapshot blob identity or size differs: ' + name)
            result[name] = set(blob['hosts'])
            if physical_paths is not None:
                physical_paths[name] = physical
    if result.keys() != files.keys():
        raise ValueError('managed file inventory is incomplete')
    refs = repo / 'refs'
    for directory, expected in [(refs, {'keep', 'main'}), (refs / 'keep', {Path(keep_ref).name})]:
        entries = call('ls', directory)['entries']
        if {row['name'] for row in entries} != expected:
            raise ValueError('managed refs differ from pinned publication')
        for row in entries:
            if row['name'] == 'keep' and row['kind'] == 'directory':
                continue
            name = (directory / row['name']).relative_to(repo).as_posix()
            if (row['kind'] != 'regular' or row['size'] != len(revision) or row.get('writing_on') is not None
                    or not row['hosts'] or (Path(managed) / model / name).read_bytes() != revision.encode()):
                raise ValueError('managed ref differs from pinned publication: ' + name)
            result[name] = set(row['hosts'])
    return result


def _check_layout(holders, files, pins, adoption_host=ADOPTION_HOST):
    dist = dict.fromkeys(nest.TARGETS, 0)
    counted = set()
    for name, hosts in holders.items():
        placed = hosts - {adoption_host}
        if name in pins:
            if placed != {PIN_HOST}:
                raise ValueError('FP8 Engram pin differs: ' + name)
        else:
            if len(placed) != 1 or not placed <= TARGETS:
                raise ValueError('managed one-copy placement differs: ' + name)
            if name.endswith('.safetensors') and files[name]['hub_oid'] not in counted:
                counted.add(files[name]['hub_oid'])
                dist[next(iter(placed))] += files[name]['bytes']
    largest = max(files[n]['bytes'] for n in files.keys() - pins if n.endswith('.safetensors'))
    if not all(dist.values()) or max(dist.values()) - min(dist.values()) > largest:
        raise ValueError('managed non-pin shards fail eight-way balance')
    return dist


def _check_rule(row, selector, size):
    if (row['selector'] != selector or row['hosts'] != [PIN_HOST] or row.get('auto') is not True
            or row.get('summary', {}).get('files') != 1
            or row['summary'].get('bytes') != size
            or row['summary'].get('models') != [TARGET_REPO]):
        raise ValueError('Engram copy rule differs from the published file: ' + row['name'])


def _pin(call, root, state_path, state, files, pins, revision, holders):
    snapshot = nest.NAMESPACE / MODEL / 'snapshots' / revision
    rules = {row['name']: row for row in call('rule', 'ls')['rules']}
    desired = {}
    for filename in sorted(pins):
        name = f'hy4-flash-{filename.removesuffix(".safetensors")}-{revision}'
        selector = str(snapshot / filename)
        if name in rules:
            _check_rule(rules[name], selector, files[filename]['bytes'])
        else:
            call('rule', 'set', name, selector, '--hosts', PIN_HOST, '--auto')
        desired[name] = filename
    rules = {row['name']: row for row in call('rule', 'ls')['rules']}
    for name, filename in desired.items():
        _check_rule(rules[name], str(snapshot / filename), files[filename]['bytes'])
    atomic_json(root / 'engram-pins.json', {name: rules[name] for name in desired})
    # Only reconcile the named rules that still need their one required copy.
    # Keep the asynchronous job ids so a resumed handoff waits on existing work.
    def one_job(*args):
        result = call(*args)
        if args[0] != 'reconcile':
            return result
        jobs = result.get('jobs')
        if not isinstance(jobs, list) or len(jobs) != 1:
            raise ValueError('unexpected native Engram reconcile job list')
        return {'job': jobs[0]}

    for name, filename in desired.items():
        key = name + '-job'
        if key not in state and key + '-submission' not in state and PIN_HOST in holders[filename]:
            continue
        nest.run_job(root, state_path, state, one_job, key, ('reconcile', name),
                     job_description=f'replicate {snapshot / filename}')


def _project(plan, holders, files, physical_paths=None, model=MODEL):
    projected = {name: set(hosts) for name, hosts in holders.items()}
    aliases = {}
    local_blobs = nest.NAMESPACE / model / 'blobs'
    for name, row in files.items():
        physical = physical_paths[name] if physical_paths is not None else local_blobs / row['hub_oid']
        aliases.setdefault(physical, []).append(name)
    seen = set()
    for step in plan['steps']:
        host = step['host']
        for row in step['copies']:
            if row.get('skip'):
                continue
            path = Path(row['path'])
            if (path, host) in seen:
                raise ValueError('native placement has duplicate operations on one file/place')
            seen.add((path, host))
            names = aliases.get(path)
            if names is None:
                names = [path.relative_to(nest.NAMESPACE / model).as_posix()]
            for name in names:
                if step['kind'] == 'replicate':
                    projected[name].add(host)
                else:
                    projected[name].remove(host)
    return projected


def _place(call, root, state_path, state, revision, files, pins, inventory, physical_paths, layout):
    key = 'placement'
    goal = ('--place', 'hf:' + layout.target.repo_id, '--factor', '1/8',
            *(v for host in nest.TARGET_ARGS for v in ('--host', host)))
    checks = dict(repo_id=layout.target.repo_id, evict_hosts=TARGETS, physical_paths=physical_paths, keep_ref=layout.keep_ref)
    proposed = root / (key + '-proposed.json')
    raw = hub.load(proposed) if proposed.exists() else call('plan', *goal)
    nest.check_plan(raw, revision, files, **checks)
    if type(raw.get('id')) is not int or raw['id'] <= 0:
        raise ValueError('native placement plan id missing')
    if not proposed.exists():
        atomic_json(proposed, raw)
    job_key = key + '-job'
    reviewed = root / (key + '.json')
    expected = deepcopy(raw)
    skipped = set()
    for step in expected['steps']:
        for row in step['copies']:
            if row['name'] in pins:
                skipped.add(row['name'])
                row['skip'] = True
    operations = [(s, c) for s in expected['steps'] for c in s['copies'] if not c.get('skip')]
    def reviewed_plan(*args):
        if args[:2] == ('plan', 'apply'):
            shown = call('plan', 'show', raw['id'], '--files')
            nest.check_plan(shown, revision, files, skipped=skipped, **checks)
            if shown != expected:
                raise ValueError('native plan changes differ from the reviewed operations')
            _check_layout(_project(shown, inventory(), files, physical_paths, layout.model), files, pins, layout.adoption_host)
            atomic_json(reviewed, shown)
            if not operations:
                return None
        return call(*args)

    # The first native submission will do its own immediate show and layout
    # check. Recovery only waits if the exact named job is already recorded.
    if reviewed.exists() and hub.load(reviewed) != expected:
        raise ValueError('persisted reviewed plan differs from the proposed operations')
    if skipped and not reviewed.exists():
        call('plan', 'skip', raw['id'], *(physical_paths[n] for n in sorted(skipped)))
    command = ('plan', 'apply', raw['id'], '--wait')
    if operations:
        eviction_count = len(operations) if all(s['kind'] == 'evict' for s, _ in operations) else None
        nest.run_job(root, state_path, state, reviewed_plan, job_key, command,
                     job_description=f"apply plan {raw['id']}", eviction_files=eviction_count)
    elif not reviewed.exists():
        reviewed_plan(*command)  # Verify placement already satisfies the goal.


def handoff(root, output, stage_root, *, call=None, managed=nest.MANAGED, layout=FLASH):
    if call is None:
        call = partial(nest.native, repo_id=layout.target.repo_id)
    root, output, stage_root = Path(root).resolve(), Path(output).absolute(), Path(stage_root).absolute()
    published, identity = _publication(root, output, layout)
    files, revision = published['files'], identity['revision']
    pins = pinned_files(files) if layout.engram else set()
    if layout.engram and not pins:
        raise ValueError('published FP8 Engram shard inventory is empty')
    with (root / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / 'handoff-complete.json'
        if marker.exists():
            complete = hub.load(marker)
            if (complete.get('schema') != layout.schema
                    or complete.get('one_fabric_copy_per_file') is not True or complete.get('status') != 'passed' or complete.get('identity') != identity
                    or complete.get('stage_root') != str(stage_root)):
                raise ValueError('Hy4 handoff receipt differs')
            return complete
        state_path = root / 'handoff-state.json'
        binding = dict(identity=identity, stage_root=str(stage_root))
        state = hub.load(state_path) if state_path.exists() else binding.copy()
        if any(state.get(k) != v for k, v in binding.items()):
            raise ValueError('Hy4 handoff recovery receipt differs')
        if not state_path.exists():
            atomic_json(state_path, state)
        if 'import-job' in state or 'import-job-submission' in state:
            repo = stage_root / layout.model
        else:
            repo = nest.stage_snapshot(published, stage_root, repo_id=layout.target.repo_id, family=layout.target.family,
                                       hub_blobs=True, keep_ref=layout.keep_ref)
        nest.run_job(root, state_path, state, call, 'import-job',
                     ('hf', 'import', repo, '--offline', '--no-copy'),
                     job_description=f'hf import {repo} -> {nest.NAMESPACE}',
                     importing_files=len({r['hub_oid'] for r in files.values()}))

        physical_paths = {}
        imported = _inventory(call, managed, files, revision, physical_paths=physical_paths, model=layout.model, keep_ref=layout.keep_ref)

        def inventory():
            current_paths = {}
            holders = _inventory(call, managed, files, revision, physical_paths=current_paths, model=layout.model, keep_ref=layout.keep_ref)
            if physical_paths != current_paths:
                raise ValueError('managed physical blob paths changed during Hy4 handoff')
            return holders

        def save_inventory(filename, holders):
            atomic_json(root / filename, {name: sorted(hosts) for name, hosts in holders.items()})

        physical_receipt = root / 'placement-physical-paths.json'
        resolved = {name: str(path) for name, path in physical_paths.items()}
        if physical_receipt.exists() and hub.load(physical_receipt) != resolved:
            raise ValueError('managed physical blob paths differ from the handoff receipt')
        atomic_json(physical_receipt, resolved)
        if not (root / 'placement-before.json').exists():
            save_inventory('placement-before.json', imported)
        if pins:
            _pin(call, root, state_path, state, files, pins, revision, imported)
        holders = inventory()
        if any(PIN_HOST not in holders[n] or not holders[n] <= {PIN_HOST, layout.adoption_host} for n in pins):
            raise ValueError('FP8 Engram files are not exclusively pinned plus the adoption copy')
        _place(call, root, state_path, state, revision, files, pins, inventory, physical_paths, layout)
        placed = inventory()
        distribution = _check_layout(placed, files, pins, layout.adoption_host)
        save_inventory('placement-after.json', placed)
        record = dict(schema=layout.schema, status='passed', identity=identity,
            repo_id=layout.target.repo_id, revision=revision, files=len(files), copied_bytes=0,
            stage_root=str(stage_root), targets=list(nest.TARGETS), distribution_bytes=distribution,
            pinned_host=PIN_HOST if pins else None, pinned_files=sorted(pins), pinned_bytes=sum(files[n]['bytes'] for n in pins),
            managed_snapshot=str(Path(managed) / layout.model / 'snapshots' / revision),
            one_fabric_copy_per_file=True, adoption_host=layout.adoption_host,
            retained_adoption_files=sum(layout.adoption_host in hosts for hosts in placed.values()),
            time=datetime.now(timezone.utc).isoformat())
        atomic_json(marker, record)
        return record


def _engram_staging(root, directory, files):
    stage = Path(directory).absolute()
    if stage != root.parent / 'output' or stage.is_symlink():
        raise ValueError('standalone Engram staging must be this task’s output directory')
    if not stage.exists():
        return stage
    expected = pinned_files(files) | {name for name in files
                if re.fullmatch(r'engram-bf16-layer-(?:01|11|21|31)\.safetensors', name)}
    for path in stage.iterdir():
        hub.regular(path)
        if path.suffix in ('.json', '.jsonl', '.csv', '.log'):
            continue  # These are archived before deletion.
        if (path.name not in expected or file_identity(path) != files[path.name]['file_identity']
                or path.stat().st_dev != files[path.name]['device']):
            raise ValueError('Engram stage has unowned or unadopted payload: ' + path.name)
    return stage


def retire(root, output, *, intermediates=(), build_root=None, engram_stage=None, layout=FLASH):
    if engram_stage and not layout.engram:
        raise ValueError('full Hy4 has no Engram staging')
    root = Path(root).resolve()
    published, identity = _publication(root, Path(output).absolute(), layout)
    complete = hub.load(root / 'handoff-complete.json')
    if (complete.get('schema') != layout.schema
            or complete.get('pinned_host') != (PIN_HOST if layout.engram else None)
            or set(complete.get('pinned_files', [])) != (pinned_files(published['files']) if layout.engram else set())):
        raise ValueError('verified matching Hy4 placement is required for cleanup')
    with (root / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / 'handoff-cleanup-complete.json'
        if marker.exists():
            record = hub.load(marker)
            if record.get('status') != 'passed' or record.get('identity') != identity:
                raise ValueError('Hy4 cleanup receipt differs')
            return record
        stage = _engram_staging(root, engram_stage, published['files']) if engram_stage else None
        return nest.retire(root, intermediates=intermediates, build_root=build_root, repo_id=layout.target.repo_id,
                           family=layout.target.family, build_prefix=layout.build_prefix, verified_staging=stage,
                           placement_check='one_fabric_copy_per_file')


def run_cli(layout):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-root', type=Path)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--engram-stage', type=Path, help='retire the task’s adopted Engram-only export')
    parser.add_argument('--cleanup-request', type=Path,
                        help='JSON containing explicit intermediates and optional build_root')
    args = parser.parse_args()
    if args.cleanup:
        request = hub.load(args.cleanup_request) if args.cleanup_request else {}
        if set(request) - {'intermediates', 'build_root'}:
            parser.error('unsupported cleanup request field')
        result = retire(args.root, args.output, engram_stage=args.engram_stage, layout=layout, **request)
    else:
        stage = args.stage_root or args.root / 'handoff-stage'
        result = handoff(args.root, args.output, stage, layout=layout)
    print(json.dumps(result), flush=True)


def main():
    run_cli(FLASH)


if __name__ == '__main__':
    main()
