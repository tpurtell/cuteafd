"""Stream an independent original/K8/K6 MC package into one final directory.

The caller owns numerical state and view metadata. Entries always name an
explicit input tensor; no parent publication, network lookup or staging copy is
required. Only one bounded output shard is assembled at a time.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

import mc_package as package
from core_export import packed_specs
from kimi_k3.uniform_k2.storage import atomic_json, canonical, file_identity, read_header, verify_receipt

VIEWS = ('original', 'k8-compact', 'k6-compact')
KINDS = ('shared', *VIEWS)


def _basename(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name or any(c in name for c in '\\\0\n\r'):
        raise ValueError('unsafe package filename')
    return name


def load_source_inventory(snapshot, *, reviewed_payload_bytes, reviewed_tensor_count,
                          extra_shards=None):
    """Check reviewed payload/count, including namespaced unindexed tensors.

    The source's advisory index total is retained for provenance, but the
    reviewed byte extent and the actual physical headers must agree exactly.
    """
    snapshot = Path(snapshot).resolve()
    extras, extra_bytes = {}, 0
    for shard, prefix in (extra_shards or {}).items():
        _basename(shard)
        header, _ = read_header(snapshot / shard)
        extras[shard] = (prefix, header)
        extra_bytes += sum(v['data_offsets'][1] - v['data_offsets'][0]
                           for n, v in header.items() if n != '__metadata__')
    inv = package.inventory(snapshot, reviewed_payload_bytes=reviewed_payload_bytes-extra_bytes)
    physical = [(n, shard) for shard, header in inv['headers'].items()
                for n in header if n != '__metadata__']
    if len(physical) != len(inv['mapping']) or dict(physical) != inv['mapping']:
        raise ValueError('source physical tensor inventory differs from complete index')
    if set(extras) & set(inv['headers']):
        raise ValueError('extra source shard is already indexed')
    locations = {n: dict(shard=shard, source_name=n) for n, shard in inv['mapping'].items()}
    for shard, (prefix, header) in extras.items():
        for name, value in header.items():
            if name == '__metadata__':
                continue
            published = prefix + name
            if published in locations:
                raise ValueError('duplicate namespaced source tensor')
            locations[published] = dict(shard=shard, source_name=name)
            inv['mapping'][published] = shard
            inv['specs'][published] = (value['dtype'], value['shape'])
            inv['sizes'][published] = value['data_offsets'][1] - value['data_offsets'][0]
    measured = sum(inv['sizes'].values())
    if measured != reviewed_payload_bytes or len(locations) != reviewed_tensor_count:
        raise ValueError('source header count or payload differs from reviewed recipe')
    return dict(mapping=inv['mapping'], locations=locations, specs=inv['specs'], sizes=inv['sizes'],
                payload_bytes=measured, tensor_count=len(locations),
                payload_totals=dict(declared=inv['index'].get('metadata', {}).get('total_size'),
                                    measured=measured, reviewed=reviewed_payload_bytes))


class DirectMCWriter:
    """Receipt-backed streaming writer; completed groups survive input retirement.

    Entry mappings use ``published_name: (input_path, input_tensor_name)``.
    ``shared`` is present in every index; each other mapping is view-specific.
    A successful group receipt is the durability frontier, so callers can retire
    only their transient encode files after ``write_group`` returns.
    """

    def __init__(self, output, source_snapshot, source_repo, source_revision, identity,
                 source_inventory, *, shard_target=2_000_000_000):
        self.output = Path(output).resolve()
        self.source = Path(source_snapshot).resolve()
        if (self.output == self.source or self.output in self.source.parents
                or self.source in self.output.parents):
            raise ValueError('source and output directories must be disjoint')
        if type(shard_target) is not int or shard_target <= 0:
            raise ValueError('shard target must be positive')
        self.shard_target = shard_target
        # Canonicalize tuples and require a caller-reviewed complete inventory.
        self.source_inventory = json.loads(canonical({k: source_inventory[k] for k in
            ('mapping', 'locations', 'specs', 'sizes', 'payload_bytes', 'tensor_count', 'payload_totals')}))
        inv = self.source_inventory
        if (not inv['mapping'] or any(set(inv[k]) != set(inv['mapping']) for k in ('locations', 'specs', 'sizes'))
                or len(inv['mapping']) != inv['tensor_count'] or sum(inv['sizes'].values()) != inv['payload_bytes']
                or inv['payload_totals']['measured'] != inv['payload_bytes']
                or inv['payload_totals']['reviewed'] != inv['payload_bytes']):
            raise ValueError('source inventory is not the reviewed complete inventory')
        for name, location in inv['locations'].items():
            if set(location) != {'shard', 'source_name'} or location['shard'] != inv['mapping'][name]:
                raise ValueError('source alias inventory differs')
            _basename(location['shard'])
        identity = dict(deepcopy(identity), schema='cuteafd-core-package-v1', family='inkling2',
                        source_repo=source_repo, source_revision=source_revision)
        self.identity = json.loads(canonical(identity))
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / 'direct-write-state.json'
        immutable = dict(identity=self.identity, source_snapshot=str(self.source),
                         source_inventory=self.source_inventory, shard_target=shard_target)
        if self.state_path.exists():
            self.state = package.load_json(self.state_path)
            if any(self.state.get(k) != v for k, v in immutable.items()):
                raise ValueError('direct writer recovery identity or source inventory differs')
        else:
            if any(self.output.iterdir()):
                raise ValueError('new direct output directory must be empty')
            self.state = dict(schema='cuteafd-direct-mc-writer-v1', **immutable, groups={})
            atomic_json(self.state_path, self.state)

    def group_complete(self, label):
        """Check committed output stat receipts without reopening retired inputs."""
        group = self.state['groups'].get(label)
        if not group or not group.get('complete'):
            return False
        for shard in group['shards']:
            verify_receipt(self.output / shard['filename'], shard['receipt'])
        return True

    def write_group(self, label, *, shared=None, original=None, k8=None, k6=None):
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', label):
            raise ValueError('unsafe direct group label')
        entries = dict(zip(KINDS, (shared or {}, original or {}, k8 or {}, k6 or {})))
        references = {kind: {n: [str(Path(p).resolve()), source_name]
                            for n, (p, source_name) in values.items()} for kind, values in entries.items()}
        previous = self.state['groups'].get(label)
        if previous and previous['entries'] != references:
            raise ValueError('direct group recovery entries differ')
        if self.group_complete(label):
            return previous
        used = {kind: set() for kind in KINDS}
        for other_label, group in self.state['groups'].items():
            if other_label != label:
                for kind in KINDS:
                    used[kind].update(group['entries'][kind])
        for kind in KINDS:
            names = set(references[kind])
            collision = used[kind] | (set().union(*(used[k] for k in VIEWS)) if kind == 'shared'
                                      else used['shared'])
            if names & collision or (kind != 'shared' and names & references['shared'].keys()):
                raise ValueError('duplicate active direct tensor')
        headers, specs, sizes, inputs = {}, {}, {}, {}
        for kind, values in references.items():
            specs[kind], sizes[kind] = {}, {}
            for name, (path, source_name) in values.items():
                if not name or any(c in name for c in '\0\n\r'):
                    raise ValueError('invalid published tensor name')
                p = Path(path)
                if self.output == p or self.output in p.parents:
                    raise ValueError('direct input cannot reference output')
                if path not in headers:
                    headers[path], _ = read_header(p)
                    inputs[path] = file_identity(p)
                value = headers[path][source_name]
                specs[kind][name] = [value['dtype'], value['shape']]
                sizes[kind][name] = value['data_offsets'][1] - value['data_offsets'][0]
                if kind == 'original':
                    location = self.source_inventory['locations'].get(name)
                    if (not location or p != (self.source / location['shard']).resolve()
                            or source_name != location['source_name']
                            or specs[kind][name] != self.source_inventory['specs'][name]):
                        raise ValueError('original tensor differs from reviewed source alias')
        if not inputs:
            raise ValueError('direct group cannot be empty')
        if previous and (previous['inputs'] != inputs or previous['specs'] != specs):
            raise ValueError('direct group input changed before completion')
        group = dict(entries=references, inputs=inputs, specs=specs, sizes=sizes, shards=[], complete=False)
        self.state['groups'][label] = group
        atomic_json(self.state_path, self.state)
        for kind, values in references.items():
            for number, names in enumerate(package.shard_groups(values, sizes[kind], self.shard_target), 1):
                filename = f'{kind}-{label}-{number:04d}.safetensors'
                receipt = package.write_shard(self.output, filename, {n: values[n] for n in names},
                    dict(producer=self.identity, group=label, kind=kind,
                         inputs={p: inputs[p] for p in sorted({values[n][0] for n in names})}),
                    prove_original=kind == 'original', verify_checksum=False)
                group['shards'].append(dict(filename=filename, kind=kind, names=names, receipt=receipt))
                atomic_json(self.state_path, self.state)
        # Detect replacement/truncation of a producer input during the write.
        if any(file_identity(p) != value for p, value in inputs.items()):
            raise ValueError('direct group input changed during assembly')
        group['complete'] = True
        atomic_json(self.state_path, self.state)
        return group

    def finalize(self, *, core_selection, configs, quantizes, assets=None,
                 expected_payload_bytes=None, verification='streaming-receipts'):
        """Seal indexes and metadata, then validate headers and durable copy proofs.

        ``core_selection`` maps module names (including ``mtp.*``) to logical
        geometry. ``configs`` and ``quantizes`` each contain the three VIEWS;
        callers explicitly choose target-head policy and retained-native tables.
        By default validation reads headers/MCG markers and checks stat-bound
        streaming write proofs, avoiding a second read of all model payloads.
        Explicit ``verification='full'`` retains independent readback auditing.
        """
        if set(configs) != set(VIEWS) or set(quantizes) != set(VIEWS) or not core_selection:
            raise ValueError('finalization requires explicit metadata for all three views')
        mappings, sizes, specs = ({k: {} for k in KINDS} for _ in range(3))
        shards = {}
        for label, group in self.state['groups'].items():
            if not self.group_complete(label):
                raise ValueError('cannot finalize an incomplete direct group')
            for row in group['shards']:
                filename, kind = row['filename'], row['kind']
                shards[filename] = row['receipt']
                for name in row['names']:
                    mappings[kind][name] = filename
                    sizes[kind][name] = group['sizes'][kind][name]
                    specs[kind][name] = group['specs'][kind][name]
        expected = {'original': {module + '.weight': ['BF16', [g['out_features'], g['in_features']]]
                                  for module, g in core_selection.items()}}
        for view, bits in (('k8-compact', 8), ('k6-compact', 6)):
            expected[view] = {n: [dtype, shape] for m, g in core_selection.items()
                              for n, (dtype, shape) in packed_specs(m, dict(g, bits=bits)).items()}
        for view in VIEWS:
            if specs[view] != expected[view]:
                raise ValueError('direct core view inventory differs from selected geometry: ' + view)
        if set(p.name for p in self.output.glob('*.safetensors')) != set(shards):
            raise ValueError('unowned physical weights in direct package')
        payload = sum(sum(v.values()) for v in sizes.values())
        if expected_payload_bytes is not None and payload != expected_payload_bytes:
            raise ValueError('physical package payload differs from reviewed recipe')
        metadata_files = []
        core_names, selected_specs, selected_mappings = {}, {}, {}
        for view in VIEWS:
            index, config, quantize, _ = package.mc_manifest.FILES[view]
            mapping = mappings['shared'] | mappings[view]
            selected_specs[view] = specs['shared'] | specs[view]
            selected_mappings[view] = mapping
            core_names[view] = sorted(mappings[view])
            atomic_json(self.output / index, dict(metadata=dict(total_size=sum(sizes['shared'].values())
                + sum(sizes[view].values())), weight_map=mapping))
            atomic_json(self.output / config, configs[view])
            atomic_json(self.output / quantize, quantizes[view])
            metadata_files.extend((index, config, quantize))
        reserved = set(metadata_files) | set(shards) | {'package-state.json', 'package-validation.json',
                    'package-provenance.json', 'direct-write-state.json', package.mc_manifest.MANIFEST}
        for name, source_path in (assets or {}).items():
            if not isinstance(name, str) or Path(name).is_absolute() or not Path(name).parts:
                raise ValueError('unsafe package asset path')
            for component in Path(name).parts:
                _basename(component)
            if Path(name).as_posix() != name:
                raise ValueError('asset path must be canonical and relative')
            if (name in reserved or name.endswith(('.safetensors', '.receipt.json'))
                    or Path(source_path).stat().st_size > 64 * 1024 * 1024):
                raise ValueError('assets must be small non-weight package files')
            destination = self.output / name
            if any(parent.is_symlink() for parent in destination.parents
                   if parent == self.output or self.output in parent.parents):
                raise ValueError('refuse asset directory symlink')
            if destination.is_symlink():
                raise ValueError('refuse asset output symlink')
            if destination.exists() and destination.read_bytes() != Path(source_path).read_bytes():
                raise ValueError('existing package asset differs')
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination)
            metadata_files.append(name)
        manifest = package.mc_manifest.build(self.output, source_repo=self.identity['source_repo'],
            source_revision=self.identity['source_revision'], core_tensors=core_names)
        atomic_json(self.output / package.mc_manifest.MANIFEST, manifest)
        metadata_files.append(package.mc_manifest.MANIFEST)
        state = dict(identity=self.identity, source_snapshot=str(self.source),
            source_tensor_locations={n: self.source_inventory['locations'][n] for n in core_names['original']},
            direct_core_selection=deepcopy(core_selection), direct_expected_specs=selected_specs,
            direct_expected_mappings=selected_mappings, direct_payload_bytes=payload,
            source_payload_bytes=self.source_inventory['payload_totals'], core_tensors=core_names,
            copies={}, new_shards=shards, new_files=sorted(set(metadata_files) | set(shards)))
        atomic_json(self.output / 'package-state.json', state)
        atomic_json(self.output / 'package-provenance.json', dict(identity=self.identity,
            source_payload_bytes=self.source_inventory['payload_totals'], physical_payload_bytes=payload,
            upstream_activations='k8-compact', views=list(VIEWS), parent_package=None))
        return package.validate(self.output, verification=verification)
