"""Resumable package publication; invoke only after TJ approves this exact tree.

No auto-publish hook exists. CPU assembly never imports or calls this module.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path

from kimi_k3.uniform_k2.storage import atomic_json, sha256_file


DESTINATIONS = {
    'glmf': 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC',
    'glmf-k325': 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-MC',
    'muse': 'wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-MC',
    'kimi-k31': 'wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC',
    'ml4': 'wrldsuksgo2mars/Mistral-Large-4-EXL3-K4',
    'step5': 'wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC',
    'glm5': 'wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC',
}


def publication_inventory(output):
    """Pure local approval packet; does not contact HF or preupload bytes."""
    from mc_package import load_json, PARENTS
    output = Path(output)
    state = load_json(output / 'package-state.json')
    validation = load_json(output / 'package-validation.json')
    if validation['identity'] != state['identity']:
        raise ValueError('validation identity differs from package')
    family = state['identity']['family']
    new = sorted(state['new_files'] + ['package-validation.json', 'package-provenance.json'])
    expected_hashes = validation['new_file_sha256']
    files = {}
    for name in new:
        path = output / name
        digest = sha256_file(path)
        if name in expected_hashes and digest != expected_hashes[name]:
            raise ValueError('validated new file changed: ' + name)
        files[name] = dict(bytes=path.stat().st_size, sha256=digest)
    parent_name = PARENTS[family][0] if family in PARENTS else 'Kimi-K3.1-Pro-EXL3-K2-Core-K6'
    parent_repo = 'wrldsuksgo2mars/' + parent_name
    copied = {}
    for target, source in sorted(state['copies'].items()):
        path = output / target
        digest = sha256_file(path)
        validated = validation['copied_file_sha256'].get(target)
        if validated is not None and digest != validated:
            raise ValueError('validated inherited file changed: ' + target)
        size = path.stat().st_size
        git_blob = None
        if size <= 64 << 20:
            raw = path.read_bytes()
            git_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        copied[target] = dict(bytes=size, sha256=digest, git_blob=git_blob)
    if family == 'ml4':
        from mc_package import family_recipe, read_header, SUFFIXES
        _, selection = family_recipe(family)
        forbidden = {m + '.' + suffix for m in selection for suffix in SUFFIXES}
        for name in list(files) + list(copied):
            if name.endswith('.safetensors') and forbidden.intersection(read_header(output / name)[0]):
                raise ValueError('ML4 publication physically contains quantized core: ' + name)
    packet = dict(schema='cuteafd-package-upload-plan-v1', repo_id=DESTINATIONS[family],
        identity=state['identity'], validation_sha256=sha256_file(output / 'package-validation.json'),
        parent_repo=parent_repo, parent_revision=state['identity']['parent_revision'],
        copies=state['copies'], copy_files=copied, new_files=files,
        new_bytes=sum(r['bytes'] for r in files.values()),
        copied_bytes=sum(r['bytes'] for r in copied.values()))
    return packet


def verified_remote_tree(api, repo, commit, expected, attributes):
    """Keep content exact, allowing only Hub-added literal LFS attributes."""
    from huggingface_hub import hf_hub_download
    from upload_model import _remote

    actual = _remote(api, repo, commit)
    if actual == expected:
        return actual, None
    name = '.gitattributes'
    if (actual.keys() != expected.keys() or name not in expected
            or any(actual[n] != row for n, row in expected.items() if n != name)):
        raise ValueError('published inventory differs from approved tree')
    # The Hub adds per-file LFS rules during commit creation. Never allow
    # rewritten original rules, arbitrary patterns, or changes to other files.
    if not attributes.endswith(b'\n') or actual[name]['bytes'] > len(attributes) + 4096 * len(expected):
        raise ValueError('unexpected server-normalized attributes')
    raw = Path(hf_hub_download(repo, name, revision=commit, token=api.token)).read_bytes()
    blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    if actual[name] != dict(bytes=len(raw), blob=blob) or not raw.startswith(attributes):
        raise ValueError('unexpected server-normalized attributes')
    added = raw[len(attributes):].splitlines(keepends=True)
    allowed = {n.encode() + b' filter=lfs diff=lfs merge=lfs -text\n'
               for n, row in expected.items() if len(row['blob']) == 64
               and not any(c in n for c in '*?[]\\\r\n\t ')}
    if not added or len(added) != len(set(added)) or not set(added) <= allowed:
        raise ValueError('unexpected server-normalized attributes')
    delta = dict(approved=expected[name], published=actual[name],
                 added_lfs_rules=[line.decode('utf-8').rstrip('\n') for line in added])
    return actual, delta


def publish(output, state_root, approved_packet, *, api=None):
    """Publish exactly the separately reviewed packet. Never call before approval."""
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub.utils._runtime import is_xet_available
    from upload_model import _remote, _restored_operation
    from write_export import _fingerprint

    if importlib.metadata.version('huggingface_hub') != '1.26.1' or not is_xet_available():
        raise ValueError('publication requires qualified huggingface_hub 1.26.1 with Xet')
    output, state_root = Path(output).resolve(), Path(state_root).resolve()
    if output == state_root or output in state_root.parents or state_root in output.parents:
        raise ValueError('upload state must be disjoint from package')
    packet = publication_inventory(output)
    if packet != approved_packet:
        raise ValueError('current package differs from separately approved upload packet')
    api = api or HfApi()
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / 'upload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = state_root / 'plan.json'
        if plan.exists() and json.loads(plan.read_text()) != packet:
            raise ValueError('resumable upload plan differs')
        atomic_json(plan, packet)
        repo = packet['repo_id']
        parent_remote = _remote(api, packet['parent_repo'], packet['parent_revision'])
        if not set(packet['copies'].values()) <= parent_remote.keys():
            raise ValueError('copy source file absent from pinned parent')
        for target, source in packet['copies'].items():
            remote, local = parent_remote[source], packet['copy_files'][target]
            if remote['bytes'] != local['bytes'] or remote['blob'] not in (local['sha256'], local['git_blob']):
                raise ValueError('local inherited bytes differ from pinned parent: ' + target)
        guard_path = state_root / 'destination.json'
        if guard_path.exists():
            guard = json.loads(guard_path.read_text())
        else:
            try:
                info = api.repo_info(repo, repo_type='model')
            except RepositoryNotFoundError:
                api.create_repo(repo, repo_type='model', private=False)
                info = api.repo_info(repo, repo_type='model')
            if info.private or set(_remote(api, repo, info.sha)) - {'.gitattributes'}:
                raise ValueError('destination must be new, empty and public')
            guard = dict(commit=info.sha)
            atomic_json(guard_path, guard)
        info = api.repo_info(repo, repo_type='model')
        if info.private:
            raise ValueError('destination became private')
        if info.sha != guard['commit'] and not (state_root / 'upload-complete.json').exists():
            # A lost commit reply is recoverable only after every preupload receipt exists.
            records = state_root / 'preuploaded'
            if any(not (records / (hashlib.sha256(n.encode()).hexdigest() + '.json')).exists()
                   for n in packet['new_files']):
                raise ValueError('destination moved before preparation completed')
        operations, expected = [], {}
        for target, source in sorted(packet['copies'].items()):
            operations.append(CommitOperationCopy(src_path_in_repo=source, path_in_repo=target,
                src_revision=packet['parent_revision'], src_repo_id=packet['parent_repo'], src_repo_type='model'))
            expected[target] = parent_remote[source]
        uploads = state_root / 'preuploaded'
        uploads.mkdir(exist_ok=True)
        for name, planned in sorted(packet['new_files'].items()):
            path = output / name
            local = _fingerprint(path)
            if local['bytes'] != planned['bytes'] or sha256_file(path) != planned['sha256']:
                raise ValueError('upload source changed after approval: ' + name)
            record_path = uploads / (hashlib.sha256(name.encode()).hexdigest() + '.json')
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['local'] != local:
                    raise ValueError('preuploaded file changed: ' + name)
                operation = _restored_operation(name, str(path), record)
            else:
                operation = CommitOperationAdd(name, str(path))
                api.preupload_lfs_files(repo, [operation], repo_type='model', revision='main',
                                       num_threads=1, free_memory=False, gitignore_content='')
                mode = operation._upload_mode
                if operation._should_ignore or mode not in ('lfs', 'regular'):
                    raise ValueError('unexpected upload classification')
                if mode == 'lfs':
                    if not operation._is_uploaded or not operation.upload_info.is_hashed:
                        raise ValueError('preupload was not acknowledged')
                    blob = operation.upload_info.sha256.hex()
                    if blob != planned['sha256']:
                        raise ValueError('uploaded LFS checksum differs')
                else:
                    if planned['bytes'] > 64 << 20:
                        raise ValueError('unexpected large regular upload')
                    raw = path.read_bytes()
                    blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
                record = dict(mode=mode, bytes=planned['bytes'], blob=blob, local=local)
                atomic_json(record_path, record)
            if record['bytes'] != planned['bytes'] or (record['mode'] == 'lfs' and record['blob'] != planned['sha256']):
                raise ValueError('preupload receipt differs from approved bytes: ' + name)
            if _fingerprint(path) != local or sha256_file(path) != planned['sha256']:
                raise ValueError('upload source changed during preupload: ' + name)
            operations.append(operation)
            expected[name] = dict(bytes=record['bytes'], blob=record['blob'])
        info = api.repo_info(repo, repo_type='model')
        if info.private:
            raise ValueError('destination became private')
        attributes = (output / '.gitattributes').read_bytes()
        if info.sha != guard['commit']:
            verified_remote_tree(api, repo, info.sha, expected, attributes)
            commit = info.sha
        else:
            commit = api.create_commit(repo, operations, repo_type='model', revision='main',
                parent_commit=guard['commit'], commit_message='Publish calibrated EXL3 checkpoint with original native core').oid
        published, normalization = verified_remote_tree(api, repo, commit, expected, attributes)
        anonymous = HfApi(token=False)
        if anonymous.model_info(repo, revision=commit).sha != commit:
            raise ValueError('published commit is not anonymously readable')
        receipt = dict(schema='cuteafd-upload-receipt-v1', status='uploaded', repo_id=repo,
                       commit=commit, plan=packet, files=published,
                       server_normalized_attributes=normalization)
        atomic_json(state_root / 'upload-complete.json', receipt)
        return receipt
