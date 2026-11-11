"""Repack target K3/K4 experts with the verified original-default K4 MC core.

CPU-only assembly and independent byte audit. No Hub writes or implicit upload.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess

import mc_package as package
from kimi_k3.uniform_k2.storage import atomic_json, read_header, sha256_file

EXPERT_REPO = 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8'
EXPERT_REVISION = 'd31ebc921f1f098b4a56d37aaccb3cdef4096184'
PARENT_REVISION = package.PARENTS['glmf-k325'][1]
EXPECTED_TARGET_TENSORS = 42 * 288 * 3 * 4
TARGET = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg)$')


def target_expert(name):
    match = TARGET.fullmatch(name)
    return bool(match and 3 <= int(match[1]) <= 44)


def plan_inputs(parent, experts):
    manifest = package.mc_manifest.validate(package.load_json(parent / 'cuteafd-mc.json'), parent)
    views = {v['name']: package.inventory(parent, v['index']) for v in manifest['variants']}
    core = {v['name']: set(v['core_tensors']) for v in manifest['variants']}
    original = views['original']
    old_experts = {n for n in original['mapping'] if target_expert(n)}
    source = package.full_parent_inventory(experts)
    selected = {n for n in source['mapping'] if target_expert(n)}
    if selected != old_experts or len(selected) != EXPECTED_TARGET_TENSORS:
        raise ValueError('target expert inventory differs from complete layers 3..44')
    storage = package.load_json(experts / 'quantize_config.json')['tensor_storage']
    modules = {n.rsplit('.', 1)[0] for n in selected}
    if not modules <= storage.keys() or any(storage[m]['bits_per_weight'] not in (3, 4) for m in modules):
        raise ValueError('target experts lack K3/K4 module metadata')
    for module in modules:
        entry = storage[module]
        for name, spec in entry['stored_tensors'].items():
            if name not in selected or source['specs'][name] != (
                    next(k for k, v in package.DTYPES.items() if v == spec['torch_dtype']), spec['shape']):
                raise ValueError('mixed expert storage metadata differs from headers')
    shared = original['mapping'].keys() - core['original'] - old_experts
    for view, inv in views.items():
        if inv['mapping'].keys() - core[view] - old_experts != shared:
            raise ValueError('parent MC shared inventories differ')
    return manifest, views, core, source, selected, modules, set(shared)


def assemble(parent, experts, output, parent_audit):
    parent, experts, output = map(lambda p: Path(p).resolve(), (parent, experts, output))
    if any(output == p or output in p.parents or p in output.parents for p in (parent, experts)):
        raise ValueError('output must be disjoint from immutable inputs')
    audit = package.load_json(parent_audit)
    if audit['commit'] != PARENT_REVISION or audit['repo_id'] != 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC':
        raise ValueError('parent must be independently verified published K4 MC')
    if experts.name != EXPERT_REVISION:
        raise ValueError('expert source must be exact pinned snapshot')
    manifest, views, core, source, selected, modules, shared = plan_inputs(parent, experts)
    source_commit = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    identity = dict(schema='cuteafd-k325-mc-package-v1', family='glmf-k325', parent_revision=PARENT_REVISION,
                    expert_repo=EXPERT_REPO, expert_revision=EXPERT_REVISION, source_commit=source_commit,
                    parent_audit_sha256=sha256_file(parent_audit))
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / 'assembly-identity.json'
    if identity_path.exists() and package.load_json(identity_path) != identity:
        raise ValueError('partial assembly identity differs')
    atomic_json(identity_path, identity)
    if (output / 'package-state.json').exists():
        if package.load_json(output / 'package-state.json')['identity'] != identity:
            raise ValueError('recovery identity differs')
        return validate(output)
    copies, receipts, new, locations = {}, {}, [], {}
    common = {}

    def link(name):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != (parent / name).resolve():
                raise ValueError('inherited link differs: ' + name)
        else:
            path.symlink_to(parent / name)

    def repack(names, inv, prefix, mapping):
        for i, group in enumerate(package.shard_groups(names, inv['sizes']), 1):
            name = f'{prefix}-{i:04d}.safetensors'
            receipts[name] = package.write_shard(output, name,
                {n: (inv['root'] / inv['mapping'][n], n) for n in group}, identity)
            new.append(name)
            mapping.update({n: name for n in group})

    repack(selected, source, 'mixed-experts', common)
    locations.update({common[n] + ':' + n: dict(name=n, target=common[n], root=str(experts), shard=source['mapping'][n], kind='target-expert') for n in selected})
    original = views['original']
    # Copy only whole shards whose physical contents are exactly desired. Mixed
    # K4 expert/core shards are never linked into the clean publication tree.
    remaining = set(shared)
    for shard in sorted({original['mapping'][n] for n in shared}):
        physical = set(original['headers'][shard]) - {'__metadata__'}
        if physical <= shared:
            link(shard)
            copies[shard] = shard
            common.update({n: shard for n in physical})
            remaining -= physical
    repack(remaining, original, 'shared-native', common)
    locations.update({common[n] + ':' + n: dict(name=n, target=common[n], root=str(parent), shard=original['mapping'][n], kind='shared-parent') for n in shared})
    weights = {}
    package_storage = package.load_json(experts / 'quantize_config.json')['tensor_storage']
    for view, inv in views.items():
        mapping = {}
        if view == 'k6-compact':
            repack(core[view], inv, 'k6-core', mapping)
        else:
            for shard in sorted({inv['mapping'][n] for n in core[view]}):
                physical = set(inv['headers'][shard]) - {'__metadata__'}
                if not physical <= core[view]:
                    raise ValueError('original/K8 shard unexpectedly mixed')
                link(shard)
                copies[shard] = shard
                mapping.update({n: shard for n in physical})
        weights[view] = common | mapping
        locations.update({mapping[n] + ':' + n: dict(name=n, target=mapping[n], root=str(parent), shard=inv['mapping'][n], kind=view) for n in core[view]})
        index_file, config_file, quant_file, _ = package.mc_manifest.FILES[view]
        config = package.load_json(parent / config_file)
        quant = package.load_json(parent / quant_file)
        for module in modules:
            quant['tensor_storage'][module] = deepcopy(package_storage[module])
        quant['bits'] = config['quantization_config']['bits'] = 3.25
        quant['meta'] = dict(package_sources=dict(
            target_experts=dict(repo=EXPERT_REPO, revision=EXPERT_REVISION, layers=[3, 44]),
            core_shared_mtp=dict(repo=audit['repo_id'], revision=PARENT_REVISION)),
            historical_parent_meta=quant.get('meta'))
        sizes = inv['sizes'] | {n: source['sizes'][n] for n in selected}
        atomic_json(output / index_file, dict(metadata=dict(total_size=sum(sizes[n] for n in weights[view])), weight_map=dict(sorted(weights[view].items()))))
        atomic_json(output / config_file, config)
        atomic_json(output / quant_file, quant)
        new += [index_file, config_file, quant_file]
    combined = package.mc_manifest.build(output, source_repo=manifest['source']['repo'],
        source_revision=manifest['source']['revision'], core_tensors=core)
    atomic_json(output / 'cuteafd-mc.json', combined)
    new.append('cuteafd-mc.json')
    excluded = {'README.md', 'README.parent.md', '.gitattributes', 'package-validation.json',
                'package-provenance.json', 'quantize_config.json', 'config.json', 'cuteafd-mc.json'}
    excluded |= {n for row in package.mc_manifest.FILES.values() for n in row[:3]}
    for name in sorted(audit['files']):
        if name in excluded or (name.endswith('.safetensors') and '/' not in name):
            continue
        link(name)
        copies[name] = name
        if name.endswith('.safetensors'):
            header, _ = read_header(parent / name)
            locations.update({name + ':' + n: dict(name=n, target=name, root=str(parent), shard=name, kind='bundled-parent')
                              for n in header if n != '__metadata__'})
    card = (parent / 'README.md').read_text()
    front = card[:card.index('\n---\n', 4) + 5]
    (output / 'README.md').write_text(front + '\n# GLM 5.3 Flash EXL3 K3.25 MC\n\n'
        'Original BF16 core is the default; K8 and K6 are complete alternate views.\n'
        'Target layers 3-44 use byte-exact mixed K3/K4 experts from `' + EXPERT_REPO + '@' + EXPERT_REVISION + '`.\n'
        'The entire layer-45 MTP drafter remains K4 from the parent MC. No vendor FP8 core payload is retained.\n'
        'Core, shared native tensors and bundled media assets derive from the verified K4 MC at `' + PARENT_REVISION + '`.\n'
        'This repack makes no new serving-performance or quality claim. See package-validation.json for byte proofs.\n')
    (output / '.gitattributes').write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    new += ['README.md', '.gitattributes']
    state = dict(identity=identity, new_files=sorted(new), copies=copies, new_shards=receipts,
                 tensor_origins=locations, core_tensors={k: sorted(v) for k, v in core.items()})
    atomic_json(output / 'package-state.json', state)
    atomic_json(output / 'package-provenance.json', identity)
    return validate(output)


def validate(output):
    output = Path(output)
    state = package.load_json(output / 'package-state.json')
    for name, receipt in state['new_shards'].items():
        package.verify_receipt(output / name, receipt, checksum=True)
    package.mc_manifest.validate(package.load_json(output / 'cuteafd-mc.json'), output)
    for view in state['core_tensors']:
        index, _, quant_file, _ = package.mc_manifest.FILES[view]
        inv = package.inventory(output, index)
        storage = package.load_json(output / quant_file)['tensor_storage']
        for entry in storage.values():
            for name, spec in entry['stored_tensors'].items():
                dtype = next(k for k, v in package.DTYPES.items() if v == spec['torch_dtype'])
                if inv['specs'].get(name) != (dtype, spec['shape']):
                    raise ValueError('view metadata differs from physical tensor: ' + name)
    physical = {}
    for path in output.rglob('*.safetensors'):
        header, _ = read_header(path)
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            if spec['dtype'].startswith('F8') or name.endswith('.weight_scale_inv'):
                raise ValueError('physical vendor FP8 payload: ' + name)
            key = path.relative_to(output).as_posix() + ':' + name
            physical[key] = (path, spec)
    if physical.keys() != state['tensor_origins'].keys():
        raise ValueError('physical inventory differs from intended union')
    proofs, counts, groups = {}, {}, {}
    for name, origin in state['tensor_origins'].items():
        source = Path(origin['root']) / origin['shard']
        groups.setdefault(source, []).append(name)
    for source, names in sorted(groups.items()):
        source_header, source_base = read_header(source)
        with ExitStack() as stack:
            source_stream = stack.enter_context(source.open('rb'))
            targets = {}
            for key in sorted(names):
                path, spec = physical[key]
                origin = state['tensor_origins'][key]
                name = origin['name']
                if path not in targets:
                    header, base = read_header(path)
                    targets[path] = (stack.enter_context(path.open('rb')), base)
                target_stream, base = targets[path]
                src = source_header[name]
                if (spec['dtype'], spec['shape']) != (src['dtype'], src['shape']):
                    raise ValueError('tensor geometry differs: ' + name)
                start, end = src['data_offsets']
                left, right = spec['data_offsets']
                if end - start != right - left:
                    raise ValueError('tensor length differs: ' + name)
                source_stream.seek(source_base + start)
                target_stream.seek(base + left)
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    count = min(16 << 20, remaining)
                    raw = source_stream.read(count)
                    if len(raw) != count or raw != target_stream.read(count):
                        raise ValueError('independent tensor bytes differ: ' + name)
                    digest.update(raw)
                    remaining -= count
                proofs[key] = digest.hexdigest()
                counts[origin['kind']] = counts.get(origin['kind'], 0) + 1
    receipt = dict(schema='cuteafd-k325-mc-validation-v1', identity=state['identity'],
        status='all-expert-core-shared-bytes-exact-no-physical-vendor-fp8', counts=counts,
        tensor_sha256=proofs, new_file_sha256={n: sha256_file(output / n) for n in state['new_files']},
        copied_file_sha256={n: sha256_file(output / n) for n in state['copies']}, physical_shards=len(list(output.rglob('*.safetensors'))))
    atomic_json(output / 'package-validation.json', receipt)
    return {k: v for k, v in receipt.items() if k not in ('tensor_sha256', 'new_file_sha256', 'copied_file_sha256')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--experts', type=Path, required=True)
    parser.add_argument('--parent-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(**vars(args)), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
