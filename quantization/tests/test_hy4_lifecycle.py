from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path

import pytest

from hy4_flash import handoff, publish, recipe
from inkling2 import handoff as nest
from inkling2 import publish as hub
from kimi_k3.uniform_k2.storage import atomic_json, file_identity

REVISION = 'a' * 40
SNAPSHOT = nest.NAMESPACE / handoff.MODEL / 'snapshots' / REVISION
BLOBS = SNAPSHOT.parent.parent / 'blobs'
REF = SNAPSHOT.parent.parent / 'refs/main'
KEEP = REF.parent / handoff.KEEP_REF
PINS = tuple(f'engram-fp8-layer-{layer:02d}-part-00000.safetensors' for layer in recipe.ENGRAM_ROWS)


def test_hy4_publication_rejects_f32_only_or_changed_engram_proof(tmp_path):
    identity = dict(source_repo=recipe.BASE_REPO, source_revision=recipe.BASE_REVISION,
                    codebook="mul1", codebook_multiplier=recipe.MUL1_MULTIPLIER)
    tensors = list(recipe.ENGRAM_TABLES)
    proof = {t: dict(rows=recipe.ENGRAM_TABLES[t], reference_dtype='BF16', mismatch_count=0) for t in tensors}
    parts = [dict(file=name, rows=recipe.ENGRAM_TABLES[t], sha256='0' * 64) for name, t in zip(PINS, tensors)]
    manifest = dict(source=dict(repo=recipe.BASE_REPO, revision=recipe.BASE_REVISION),
                    original_core_dtype='BF16', exl3_codebook=dict(name='mul1', multiplier=recipe.MUL1_MULTIPLIER),
                    variants=[dict(name=n) for n in ('original', 'k8-compact', 'k6-compact')],
                    engram_fp8=dict(recipe='test; reference=' + recipe.REFERENCE_REVISION,
                                   layers=[dict(tensor=t, parts=[p]) for t, p in zip(tensors, parts)]))
    path = tmp_path / 'cuteafd-mc.json'
    atomic_json(path, manifest)
    validation = dict(new_file_sha256={path.name: hub.metadata_identity(path)['sha256']})
    state = dict(identity=identity, engram_fp8_proof=proof,
                 new_shards={p['file']: dict(sha256=p['sha256']) for p in parts})
    publish._validate_package(tmp_path, state, validation)
    proof[tensors[0]]['reference_dtype'] = 'F32'
    with pytest.raises(ValueError, match='BF16 proof'):
        publish._validate_package(tmp_path, state, validation)
    proof[tensors[0]]['reference_dtype'] = 'BF16'
    state['new_shards'][PINS[0]]['sha256'] = '1' * 64
    with pytest.raises(ValueError, match='sealed shard'):
        publish._validate_package(tmp_path, state, validation)


@pytest.fixture
def published(tmp_path):
    return publication_fixture(tmp_path, handoff.FLASH, PINS)


def publication_fixture(tmp_path, layout, pins):
    root, output, source, managed = (tmp_path / n for n in ('production', 'package', 'source', 'managed'))
    for p in (root, output, source):
        p.mkdir()
    files = {}
    for name in (*pins, *(f'weight-{i:02d}.safetensors' for i in range(8))):
        path = output / name
        contents = name.encode().ljust(100, b'0')
        path.write_bytes(contents)
        files[name] = dict(path=str(path), bytes=len(contents), sha256=sha256(contents).hexdigest(),
                           git_blob=None, hub_oid=sha256(contents).hexdigest(), file_identity=file_identity(path), device=path.stat().st_dev)
    identity = dict(family=layout.target.family)
    atomic_json(output / 'package-state.json', dict(identity=identity))
    metadata = dict(schema=layout.target.schema('complete'), status='passed',
                    repo_id=layout.target.repo_id, revision=REVISION, output=str(output),
                    source_snapshot=str(source), files=files, identity=identity,
                    manifest_sha256=hub.manifest_digest(files))
    atomic_json(root / 'publication-complete.json', metadata)
    ref = managed / layout.model / 'refs/main'
    ref.parent.mkdir(parents=True)
    ref.write_text(REVISION)
    keep = ref.parent / layout.keep_ref
    keep.parent.mkdir()
    keep.write_text(REVISION)
    return root, output, managed, metadata


class Native:
    def __init__(self, files, *, shared_blobs=False, layout=handoff.FLASH):
        self.layout = layout
        self.pins = PINS if layout.engram else ()
        self.snapshot = nest.NAMESPACE / layout.model / "snapshots" / REVISION
        self.blobs = self.snapshot.parent.parent / "blobs"
        self.ref = self.snapshot.parent.parent / "refs/main"
        self.keep = self.ref.parent / layout.keep_ref
        self.files = files
        self.shared_blobs = shared_blobs
        self.physical = {}
        for name, row in files.items():
            oid = row['hub_oid']
            key = sha256(('shared ' + oid).encode()).hexdigest()
            self.physical[name] = (nest.NAMESPACE / 'blobs' / key[:2] / key
                                   if shared_blobs else self.blobs / oid)
        self.calls = []
        self.rules = {}
        self.plans = {}
        self.placements = {name: {layout.adoption_host} for name in (*files, 'refs/main', 'refs/' + layout.keep_ref)}
        self.jobs = {}
        self.fail_wait = None
        self.change_show = False
        self.fail_submit = False

    def _job(self, count, *, what, host='maxq', importing=False, eviction=False):
        job = dict(id=len(self.jobs) + 1, what=what, finished=True, hosts={host: dict(finished=not eviction, total_files=count,
                                                   done_files=count, done_bytes=0, total_bytes=0)})
        result = dict(job=job)
        if importing:
            result['import'] = dict(finished=True, errors=[], copied_bytes=0, adopted=count)
        self.jobs[len(self.jobs) + 1] = result
        return len(self.jobs)

    def _plan(self, args):
        assert args[0] == '--place'
        goal = dict(goal='place', selector='hf:' + self.layout.target.repo_id,
                    hub=str(nest.NAMESPACE), hosts=list(nest.TARGET_ARGS), factor=1/8, by_speed=False)
        rows = {name: dict(path=str(self.physical[name]), group=self.layout.target.repo_id, name=name, size=row['bytes'])
                for name, row in self.files.items()}
        steps = [dict(kind='replicate', host=host, copies=[rows[f'weight-{i:02d}.safetensors']])
                 for i, host in enumerate(nest.TARGETS)]
        steps[0]['copies'].extend([rows[name] for name in self.pins])
        for ref in (self.ref, self.keep):
            steps[0]['copies'].append(dict(path=str(ref), size=len(REVISION), name=ref.name, group=str(ref.parent)))
        plan = dict(id=1, feasible=True, blocked=[], goal=goal, steps=steps)
        self.plans[plan['id']] = plan
        return deepcopy(plan)

    def __call__(self, *args):
        self.calls.append(args)
        match args:
            case ('hf', 'import', repo, '--offline', '--no-copy'):
                assert repo.name == self.layout.model
                return {'job': self._job(len(self.files), what=f'hf import {repo} -> {nest.NAMESPACE}', importing=True)}
            case ('jobs',):
                return dict(jobs=[result['job'] for result in self.jobs.values()])
            case ('jobs', number, '--wait'):
                if number == self.fail_wait:
                    self.fail_wait = None
                    raise TimeoutError('interrupted native job wait')
                return self.jobs[number]
            case ('rule', 'ls'):
                return dict(rules=list(self.rules.values()))
            case ('rule', 'set', name, selector, '--hosts', host, '--auto'):
                assert host == 'raptor'
                self.rules[name] = dict(name=name, selector=selector, hosts=[host], auto=True,
                    summary=dict(files=1, bytes=self.files[Path(selector).name]['bytes'], models=[self.layout.target.repo_id]))
                return {}
            case ('reconcile', rule):
                self.placements[Path(self.rules[rule]['selector']).name].add('raptor')
                return {'jobs': [self._job(1, host='raptor', what='replicate ' + self.rules[rule]['selector'])]}
            case ('plan', '--place', *_):
                return self._plan(args[1:])
            case ('plan', 'skip', number, *paths):
                assert set(paths) == {self.physical[n] for n in self.pins}
                for step in self.plans[number]['steps']:
                    for row in step['copies']:
                        if Path(row['path']) in paths:
                            row['skip'] = True
                return {}
            case ('plan', 'show', number, '--files'):
                plan = deepcopy(self.plans[number])
                if self.change_show:
                    plan['steps'][0]['copies'][0]['size'] += 1
                return plan
            case ('plan', 'apply', number, '--wait'):
                self.placements = handoff._project(self.plans[number], self.placements, self.files, self.physical, self.layout.model)
                count = sum(not row.get('skip') for step in self.plans[number]['steps'] for row in step['copies'])
                job = self._job(count, what=f'apply plan {number}', eviction=number == 2)
                if self.fail_submit:
                    self.fail_submit = False
                    raise TimeoutError('native job accepted before client lost reply')
                if self.fail_wait == job:
                    self.fail_wait = None
                    raise TimeoutError('interrupted native job wait')
                return self.jobs[job]
            case ('ls', path) if path == self.snapshot.parent:
                return dict(entries=[dict(name=REVISION, kind='directory')])
            case ('ls', path) if path == self.ref.parent:
                return dict(entries=[dict(name='main', kind='regular', size=len(REVISION),
                                          hosts=sorted(self.placements['refs/main'])), dict(name='keep', kind='directory')])
            case ('ls', path) if path == self.keep.parent:
                return dict(entries=[dict(name=self.keep.name, kind='regular', size=len(REVISION),
                                          hosts=sorted(self.placements['refs/' + self.layout.keep_ref]))])
            case ('ls', path) if path == self.snapshot:
                return dict(entries=[dict(name=name, kind='symlink', target='../../blobs/' + row['hub_oid'])
                                     for name, row in self.files.items()])
            case ('ls', path) if path == self.blobs:
                if self.shared_blobs:
                    return dict(entries=[dict(name=row['hub_oid'], kind='symlink',
                                              target=os.path.relpath(self.physical[name], self.blobs))
                                         for name, row in self.files.items()])
                return dict(entries=[dict(name=row['hub_oid'], kind='regular', sealed=True, size=row['bytes'],
                                          hosts=sorted(self.placements[name])) for name, row in self.files.items()])
            case ('ls', path) if self.shared_blobs:
                name = next(n for n, physical in self.physical.items() if physical == path)
                return dict(entries=[dict(name=path.name, kind='regular', sealed=True,
                                          size=self.files[name]['bytes'], hosts=sorted(self.placements[name]))])
            case _:
                pytest.fail('unexpected simulated SparkNest operation ' + repr(args))


@pytest.mark.parametrize("shared_blobs", [False, True])
def test_hy4_no_copy_import_pin_spread_resume_and_cleanup(published, monkeypatch, shared_blobs):
    root, output, managed, record = published
    call = Native(record['files'], shared_blobs=shared_blobs)
    call.fail_wait = len(PINS) + 2  # pause after the native spread job has been submitted
    stage = root / 'handoff-stage'
    with pytest.raises(TimeoutError, match='interrupted'):
        handoff.handoff(root, output, stage, call=call, managed=managed)
    source = output / PINS[0]
    engram_stage = root.parent / 'output'
    engram_stage.mkdir()
    os.link(source, engram_stage / source.name)
    (engram_stage / 'engram-export-complete.json').write_text('{}')
    assert source.samefile(stage / handoff.MODEL / 'snapshots' / REVISION / source.name)
    result = handoff.handoff(root, output, stage, call=call, managed=managed)
    assert (stage / handoff.MODEL / 'blobs' / record['files'][source.name]['hub_oid']).samefile(source)
    assert (stage / handoff.MODEL / 'snapshots' / REVISION / source.name).is_symlink()
    assert (stage / handoff.MODEL / 'refs' / handoff.KEEP_REF).read_bytes() == REVISION.encode()
    assert result['retained_adoption_files'] == len(record['files']) + 2
    assert (root / 'placement-before.json').is_file() and (root / 'placement-after.json').is_file()
    assert result['pinned_host'] == 'raptor'
    assert result['pinned_files'] == sorted(PINS)
    assert set(result['distribution_bytes'].values()) == {100}
    assert [args for args in call.calls if args[:2] == ('plan', 'apply')] == [
        ('plan', 'apply', 1, '--wait')]
    assert [args for args in call.calls if args[:2] == ('hf', 'import')] == [
        ('hf', 'import', stage / handoff.MODEL, '--offline', '--no-copy')]
    assert all(call.placements[name] == {'raptor', 'maxq'} for name in PINS)
    assert output.is_dir() and stage.is_dir()
    original_retire = nest.retire
    monkeypatch.setattr(nest, 'retire', lambda *a, **kw: original_retire(*a, **kw, inspect=lambda paths: None))
    retired = handoff.retire(root, output, engram_stage=engram_stage)
    assert handoff.retire(root, output, engram_stage=engram_stage) == retired
    assert retired['status'] == 'passed'
    assert not output.exists() and not stage.exists() and not engram_stage.exists()
    assert (root / 'retired-receipts/output/package-state.json').exists()
    assert (root / 'retired-receipts/verified-staging/engram-export-complete.json').exists()
    before = list(call.calls)
    assert handoff.handoff(root, output, stage, call=call, managed=managed) == result
    assert call.calls == before


def test_hy4_changed_native_plan_stops_before_apply(published):
    root, output, managed, record = published
    call = Native(record['files'])
    call.change_show = True
    with pytest.raises(ValueError, match='unreviewed file'):
        handoff.handoff(root, output, root / 'stage', call=call, managed=managed)
    assert not any(args[:2] == ('plan', 'apply') for args in call.calls)
    assert not (root / 'handoff-complete.json').exists()
    assert output.exists()


def test_hy4_recovers_plan_accepted_before_reply(published):
    root, output, managed, record = published
    call = Native(record['files'])
    call.fail_submit = True
    stage = root / 'stage'
    with pytest.raises(TimeoutError, match='lost reply'):
        handoff.handoff(root, output, stage, call=call, managed=managed)
    assert 'placement-job' not in hub.load(root / 'handoff-state.json')
    assert handoff.handoff(root, output, stage, call=call, managed=managed)['status'] == 'passed'
    assert call.calls.count(('plan', 'apply', 1, '--wait')) == 1


def test_hy4_cleanup_protects_unrelated_engram_staging(published):
    root, output, managed, record = published
    call = Native(record['files'])
    handoff.handoff(root, output, root / 'stage', call=call, managed=managed)
    engram_stage = root.parent / 'output'
    engram_stage.mkdir()
    (engram_stage / 'unrelated.safetensors').write_bytes(b'not an adopted hardlink')
    with pytest.raises(ValueError, match='unowned or unadopted'):
        handoff.retire(root, output, engram_stage=engram_stage)
    assert output.exists() and engram_stage.exists()


def test_stage_binds_nested_aliases_to_one_hub_git_oid(published):
    root, output, _, record = published
    for name in ('config.json', 'core/config.json'):
        path = output / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('{}')
        hashes = hub.metadata_identity(path)
        record['files'][name] = dict(path=str(path), bytes=2, device=path.stat().st_dev,
            file_identity=file_identity(path), hub_oid=hashes['git_blob'], **hashes)
    record['manifest_sha256'] = hub.manifest_digest(record['files'])
    stage = root / 'nested-stage'
    repo = nest.stage_snapshot(record, stage, repo_id=publish.TARGET_REPO, family=publish.FAMILY,
                               hub_blobs=True, keep_ref=handoff.KEEP_REF)
    snapshot = repo / 'snapshots' / REVISION
    assert (snapshot / 'config.json').samefile(snapshot / 'core/config.json')
    assert (snapshot / 'core/config.json').readlink() == Path('../../../blobs') / hashes['git_blob']
    assert len(list((repo / 'blobs').iterdir())) == len(record['files']) - 1
    holders = {'config.json': {'maxq'}, 'core/config.json': {'maxq'}}
    step = dict(host='emu', kind='replicate', copies=[dict(path=str(BLOBS / hashes['git_blob']))])
    after = handoff._project({'steps': [step]}, holders, record['files'])
    assert list(after.values()) == [{'maxq', 'emu'}, {'maxq', 'emu'}]
    (snapshot / 'core/config.json').unlink()
    (snapshot / 'core/config.json').symlink_to(output / 'config.json')
    with pytest.raises(ValueError, match='snapshot link differs'):
        nest.stage_snapshot(record, stage, repo_id=publish.TARGET_REPO, family=publish.FAMILY,
                            hub_blobs=True, keep_ref=handoff.KEEP_REF)


def test_stage_refuses_wrong_hub_oid_before_creating_links(published):
    root, _, _, record = published
    record['files'][PINS[0]]['hub_oid'] = '0' * 64
    stage = root / 'bad-oid'
    with pytest.raises(ValueError, match='verified Hub blob OIDs'):
        nest.stage_snapshot(record, stage, repo_id=publish.TARGET_REPO, family=publish.FAMILY,
                            hub_blobs=True, keep_ref=handoff.KEEP_REF)
    assert not list(stage.glob('models--*'))


def test_hy4_refuses_adoption_eviction_before_any_apply(published):
    root, output, managed, record = published

    class EvictAdoption(Native):
        def _plan(self, args):
            plan = super()._plan(args)
            plan['steps'].append(dict(kind='evict', host='maxq', copies=[plan['steps'][0]['copies'][0]]))
            return plan

    call = EvictAdoption(record['files'])
    with pytest.raises(ValueError, match='evicts from an unauthorized disk'):
        handoff.handoff(root, output, root / 'stage', call=call, managed=managed)
    assert not any(args[:2] == ('plan', 'apply') for args in call.calls)
    assert output.is_dir() and not (root / 'handoff-complete.json').exists()

@pytest.mark.parametrize("family", ["hy4", "glm5_flash"])
def test_no_engram_placement_uses_its_own_ref_and_adoption_host(tmp_path, monkeypatch, family):
    if family == "hy4":
        from hy4.handoff import LAYOUT
    else:
        from glm5_flash.mul1_mc.publish import LAYOUT

    root, output, managed, record = publication_fixture(tmp_path, LAYOUT, ())
    native = Native(record["files"], shared_blobs=True, layout=LAYOUT)
    stage = root / "handoff-stage"
    result = handoff.handoff(root, output, stage, layout=LAYOUT, call=native, managed=managed)
    assert result["revision"] == REVISION and result["schema"] == f"cuteafd-{family}-handoff-complete-v2"
    assert result["adoption_host"] == LAYOUT.adoption_host
    assert (stage / LAYOUT.model / "refs" / LAYOUT.keep_ref).read_bytes() == REVISION.encode()
    assert result["one_fabric_copy_per_file"] and result["distribution_bytes"] == dict.fromkeys(nest.TARGETS, 100)
    assert result["pinned_files"] == [] and result["pinned_host"] is None and result["pinned_bytes"] == 0
    assert not any(args[0] in ("rule", "reconcile") for args in native.calls)
    assert [args for args in native.calls if args[:2] == ("plan", "apply")] == [("plan", "apply", 1, "--wait")]
    original_retire = nest.retire
    monkeypatch.setattr(nest, "retire", lambda *a, **kw: original_retire(*a, **kw, inspect=lambda paths: None))
    receipt = handoff.retire(root, output, layout=LAYOUT)
    assert receipt["source_and_shared_stores_preserved"]
    assert (root / "retired-receipts/output/package-state.json").is_file()
