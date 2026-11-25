"""Resumable HTTP LFS package publication; invoke only after TJ approves this exact tree.

No auto-publish hook exists. CPU assembly never imports or calls this module.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

from hub_requirements import require_qualified_hub
import hub_lfs
from kimi_k3.uniform_k2.storage import atomic_json, sha256_file, verify_receipt


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
    """Pure local approval packet; does not contact HF or preupload bytes.

    New shards reuse the writer's streaming SHA256 receipt (stat-checked), so
    approval and upload never reread their weights; metadata is hashed.
    """
    from mc_package import load_json, PARENTS
    output = Path(output)
    state = load_json(output / 'package-state.json')
    validation = load_json(output / 'package-validation.json')
    if validation['identity'] != state['identity']:
        raise ValueError('validation identity differs from package')
    family = state['identity']['family']
    new = sorted(state['new_files'] + ['package-validation.json', 'package-provenance.json'])
    expected_hashes = validation['new_file_sha256']
    receipts = state.get('new_shards', {})
    files = {}
    for name in new:
        path = output / name
        receipt = receipts.get(name, {})
        if receipt.get('hash_algorithm', 'sha256') == 'sha256' and 'sha256' in receipt:
            verify_receipt(path, receipt)  # stat only: the writer hashed its byte stream.
            digest = receipt['sha256']
        else:
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

    actual = hub_lfs.remote_tree(api, repo, commit)
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


def publish(output, state_root, approved_packet, *, api=None, options=None):
    """Publish exactly the separately reviewed packet. Never call before approval.

    Verified HTTP LFS batch commits on a guarded parent; the first batch also
    carries the server-side parent copies. Resume reads the remote tree and
    never sends a committed file again.
    """
    from huggingface_hub import CommitOperationCopy, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    require_qualified_hub()
    options = options or hub_lfs.UploadOptions()
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
        parent_remote = hub_lfs.remote_tree(api, packet['parent_repo'], packet['parent_revision'])
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
            if info.private or set(hub_lfs.remote_tree(api, repo, info.sha)) - {'.gitattributes'}:
                raise ValueError('destination must be new, empty and public')
            guard = dict(commit=info.sha)
            atomic_json(guard_path, guard)
        copies = {target: parent_remote[source] for target, source in packet['copies'].items()}
        files = {}
        for name, planned in packet['new_files'].items():
            path = output / name
            files[name] = dict(path=str(path), bytes=planned['bytes'], sha256=planned['sha256'],
                               git_blob=None if name.endswith('.safetensors') else hub_lfs.git_blob(path),
                               local=hub_lfs.fingerprint(path))
            if files[name]['local']['bytes'] != planned['bytes']:
                raise ValueError('upload source changed after approval: ' + name)
        attributes = (output / '.gitattributes').read_bytes()

        def expected_tree(tree, names):
            """Exact rows for committed ``names`` plus every server-side copy."""
            expected = dict(copies)
            for name in names:
                if name not in tree:
                    raise ValueError('published inventory differs from approved tree')
                if name == '.gitattributes' and not hub_lfs.matches(tree[name], files[name]):
                    # Hub-appended LFS rules are checked by verified_remote_tree.
                    expected[name] = dict(bytes=files[name]['bytes'], blob=files[name]['git_blob'])
                else:
                    expected[name] = hub_lfs.resolved(tree[name], files[name], name)
            return expected

        info = api.repo_info(repo, repo_type='model')
        if info.private:
            raise ValueError('destination became private')
        done = set()
        if info.sha != guard['commit']:
            # Earlier batches (or a lost commit reply) landed. The first batch carries
            # every copy and our attributes, so any other moved head is refused.
            tree = hub_lfs.remote_tree(api, repo, info.sha)
            done = hub_lfs.committed_names(tree, files, fixed=copies, ignore={'.gitattributes'})
            if not set(copies) <= set(tree) or not done or '.gitattributes' not in tree:
                raise ValueError('destination moved before the first verified batch')
            done.add('.gitattributes')
            verified_remote_tree(api, repo, info.sha, expected_tree(tree, done), attributes)
        pending = [name for name in hub_lfs.upload_order(files) if name not in done]
        leading = [] if done else [CommitOperationCopy(src_path_in_repo=source, path_in_repo=target,
            src_revision=packet['parent_revision'], src_repo_id=packet['parent_repo'], src_repo_type='model')
            for target, source in sorted(packet['copies'].items())]

        def verify(commit, batch, operations):
            done.update(batch)
            verified_remote_tree(api, repo, commit, expected_tree(hub_lfs.remote_tree(api, repo, commit), done),
                                 attributes)

        commit = info.sha
        if pending or leading:
            commit = hub_lfs.commit_batches(api, repo, files, pending, revision=info.sha, leading=leading,
                commit_message='Publish calibrated EXL3 checkpoint with original native core', options=options,
                check_sources=lambda batch: hub_lfs.check_fingerprints(files, batch), verify=verify)
        expected = expected_tree(hub_lfs.remote_tree(api, repo, commit), files)
        published, normalization = verified_remote_tree(api, repo, commit, expected, attributes)
        anonymous = HfApi(token=False)
        if anonymous.model_info(repo, revision=commit).sha != commit:
            raise ValueError('published commit is not anonymously readable')
        receipt = dict(schema='cuteafd-upload-receipt-v1', status='uploaded', repo_id=repo,
                       commit=commit, plan=packet, files=published,
                       server_normalized_attributes=normalization)
        atomic_json(state_root / 'upload-complete.json', receipt)
        return receipt


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path, help='upload state root (required to publish)')
    parser.add_argument('--approved-packet', type=Path, help='separately approved inventory JSON')
    parser.add_argument('--inventory-only', action='store_true')
    hub_lfs.add_arguments(parser)
    args = parser.parse_args()
    if args.inventory_only:
        print(json.dumps(publication_inventory(args.output)), flush=True)
        return
    if args.state is None or args.approved_packet is None:
        parser.error('publication requires --state and --approved-packet')
    receipt = publish(args.output, args.state, json.loads(args.approved_packet.read_text()),
                      options=hub_lfs.options_from(args))
    print(json.dumps({key: receipt[key] for key in ('status', 'repo_id', 'commit')}), flush=True)


if __name__ == '__main__':
    main()
