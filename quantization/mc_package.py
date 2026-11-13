"""CPU-streamed native-core restoration and strict multi-core package assembly.

This tool never creates a Hub repository or uploads. Parent shard links are a
publication plan: the publisher must copy them from the pinned parent revision.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/lib'))
import mc_manifest
from core_export import MCG, SUFFIXES, packed_specs
from kimi_k3.uniform_k2.storage import (
    assemble_shard, atomic_json, read_header, sha256_file, verify_receipt,
)

DTYPES = {'BF16': 'bfloat16', 'F32': 'float32', 'F16': 'float16',
          'F8_E4M3': 'float8_e4m3fn', 'I16': 'int16', 'I32': 'int32'}
PARENTS = {
    'glmf': ('GLM-5.3-Flash-EXL3-K4-Core-K6', '5d3fffe1df9c4a109bbbdafb60dc19df9fac019d'),
    'glmf-k325': ('GLM-5.3-Flash-EXL3-K4-MC', '91a8dbc9c12e790c65aa1a38c6aede6a3e23a77d'),
    'muse': ('Muse-Spark-Open-EXL3-K4-Core-K6', '89998fb177acd9a862ac8bd4fb79bfd50e940c97'),
    'ml4': ('Mistral-Large-4-EXL3-K4-Core-K6', '03c80aebaa483d55760703ae5bc0c9734c9d67d6'),
    'step5': ('Step-5-Preview-EXL3-K4-Core-K6', '83d47a384eabc07ea6fcfaac4779b0e5f013ce89'),
    'glm5': ('GLM-5.3-EXL3-K4-Core-K6', '5a4e83a09a11f3145332ed632d28a9894ea88225'),
}


def family_recipe(family):
    if family == 'glmf':
        from glm5_flash.core_k6 import recipe
        selection = recipe.reviewed_core_modules(include_mtp=False, core_bits=8)
    elif family == 'muse':
        from muse_spark.core_k6 import recipe
        selection = recipe.reviewed_core_modules(core_bits=8)
    elif family == 'kimi-k31':
        from kimi_k31.k2_core_k6 import recipe
        selection = recipe.reviewed_core_modules()
    elif family == 'ml4':
        from mistral_large4.core_k6 import recipe
        selection = recipe.reviewed_core_modules()
    elif family in ('step5', 'glm5'):
        if family == 'step5':
            from step5.core_k6 import recipe
        else:
            from glm5.core_k6 import recipe
        selection = recipe.reviewed_core_modules(include_mtp=False)
    else:
        raise ValueError('unknown package family')
    return recipe, selection


def load_json(path):
    return json.loads(Path(path).read_text())


def inventory(root, index_name='model.safetensors.index.json', *, reviewed_payload_bytes=None):
    root = Path(root)
    index = load_json(root / index_name)
    mapping = index['weight_map']
    headers, specs, sizes = {}, {}, {}
    for shard in sorted(set(mapping.values())):
        if Path(shard).name != shard or not shard.endswith('.safetensors'):
            raise ValueError('index shard must be a safetensors basename')
        headers[shard], _ = read_header(root / shard)
    for name, shard in mapping.items():
        value = headers[shard][name]
        specs[name] = (value['dtype'], value['shape'])
        sizes[name] = value['data_offsets'][1] - value['data_offsets'][0]
    measured = sum(sizes.values())
    if reviewed_payload_bytes is not None:
        if measured != reviewed_payload_bytes:
            raise ValueError('source header payload differs from reviewed recipe')
    elif index.get('metadata', {}).get('total_size') != measured:
        raise ValueError('selected index total_size differs from headers')
    return dict(root=root, index=index, mapping=mapping, headers=headers, specs=specs, sizes=sizes)


def full_parent_inventory(root):
    result = inventory(root)
    physical = {name: shard for shard, h in result['headers'].items()
                for name in h if name != '__metadata__'}
    count = sum(len(h) - ('__metadata__' in h) for h in result['headers'].values())
    if count != len(physical) or physical != result['mapping']:
        raise ValueError('parent physical tensor inventory differs from complete index')
    return result


def tensor_hash(path, name):
    header, base = read_header(path)
    start, end = header[name]['data_offsets']
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(base + start)
        remaining = end - start
        while remaining:
            chunk = stream.read(min(16 << 20, remaining))
            if not chunk:
                raise ValueError('truncated tensor during SHA proof')
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def restored_names(family, selection):
    recipe, _ = family_recipe(family)
    return {name for module, geometry in selection.items()
            for name in (recipe.source_names(module, geometry) if family == 'ml4'
                         else (module + '.weight',))}


def shard_groups(names, sizes, target=2_000_000_000):
    group, size = [], 0
    for name in sorted(names):
        if group and size + sizes[name] > target:
            yield group
            group, size = [], 0
        group.append(name)
        size += sizes[name]
    if group:
        yield group


def write_shard(root, filename, entries, identity):
    path = root / filename
    receipt_path = root / (filename + '.receipt.json')
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if receipt['identity'] != identity or receipt['tensors'] != sorted(entries):
            raise ValueError('package recovery identity differs')
        verify_receipt(path, receipt, checksum=True)
    else:
        if path.is_symlink():
            raise ValueError('refuse package output symlink')
        path.unlink(missing_ok=True)
        receipt = dict(assemble_shard(entries, path), identity=identity, tensors=sorted(entries))
        atomic_json(receipt_path, receipt)
    return receipt


def exact_patterns(patterns, universe, removed):
    """Retain parent matching semantics while excluding only restored modules."""
    if not removed:
        return list(patterns)
    exclusion = '(?!(?:' + '|'.join(re.escape(n) for n in sorted(removed)) + ')$)'
    result = ['^' + exclusion + '(?:' + '|'.join(patterns) + ')']
    before = {name for name in universe if any(re.search(p, name) for p in patterns)}
    after = {name for name in universe if any(re.search(p, name) for p in result)}
    if after != before - removed:
        raise ValueError('module include rewrite differs from exact selected module set')
    return result


def view_metadata(parent_quant, parent_config, selection, specs, *, bits, quantize_file):
    quant, config = deepcopy(parent_quant), deepcopy(parent_config)
    storage = quant['tensor_storage']
    if not set(selection) <= storage.keys():
        raise ValueError('reviewed core missing from parent per-module metadata')
    for module, geometry in selection.items():
        if bits is None:
            del storage[module]
        else:
            entry = storage[module]
            entry.update(bits_per_weight=bits, mcg_multiplier=MCG,
                         stored_tensors={name: dict(shape=shape, torch_dtype=DTYPES[dtype])
                            for name, (dtype, shape) in packed_specs(module, dict(geometry, bits=bits)).items()})
    if bits is None:
        quant['module_include'] = exact_patterns(quant['module_include'],
                                                parent_quant['tensor_storage'], set(selection))
    quant['lm_head'] = bits is not None
    for key in ('core_bits', 'head_bits'):
        if bits is None:
            quant.pop(key, None)
        else:
            quant[key] = bits
    if bits is None:
        if 'kept_tensors' in quant:
            quant['kept_tensors'].update({name: dict(shape=shape, torch_dtype=DTYPES[dtype])
                for name, (dtype, shape) in specs.items()})
        if 'retained_native_tensors' in quant:
            quant['retained_native_tensors'].update({name: dict(shape=shape, dtype=dtype)
                for name, (dtype, shape) in specs.items()})
    quant['per_module_metadata'] = quantize_file + ':tensor_storage'
    inline = config['quantization_config']
    inline['per_module_metadata'] = quant['per_module_metadata']
    for key in ('core_bits', 'head_bits'):
        if bits is None:
            inline.pop(key, None)
        else:
            inline[key] = bits
    return quant, config


def calibration_count_key(identity):
    keys = [key for key in ('samples', 'prompts') if key in identity['calibration']]
    if len(keys) != 1:
        raise ValueError('K8 calibration requires exactly one of samples or prompts')
    return keys[0]


def k8_diagnostics(root, selection, identity):
    """Retain every real-row reconstruction/oracle report without model imports."""
    root = Path(root)
    paths = sorted(root.rglob('*.json'))
    records = {}
    tokens = identity['calibration']['tokens']
    if identity['calibration'][calibration_count_key(identity)] != 1426:
        raise ValueError('K8 diagnostics require the complete calibration corpus')
    for path in paths:
        row = load_json(path)
        module = row.get('module')
        if module is None:
            continue
        if module not in selection or module in records:
            raise ValueError('unexpected or duplicate K8 diagnostic module')
        geometry = selection[module]
        evidence, metrics = row.get('evidence', {}), row.get('metrics', {})
        oracle = row.get('packed_byte_oracle', evidence.get('packed_byte_oracle', {}))
        reconstruction = metrics.get('reconstruction', {})
        error = metrics.get('hessian_weighted_relative_error')
        if (row.get('logical_shape') != [geometry['out_features'], geometry['in_features']]
                or evidence.get('bits') != 8 or evidence.get('identity_rows') != 0
                or evidence.get('hessian_rows') != tokens
                or metrics.get('hessian_sample_count') != tokens
                or metrics.get('quantizer_path') != 'hessian_ldlq'
                or metrics.get('hessian_numerical_contract') != 'signed-block-hadamard-congruence-fp64-v1'
                or not isinstance(error, (float, int)) or not math.isfinite(error) or error < 0
                or reconstruction.get('reference_finite') is not True
                or reconstruction.get('error_finite') is not True
                or oracle.get('passed') is not True or oracle.get('bits') != 8
                or oracle.get('schema') != 'cuteafd-core-packed-byte-oracle-v1'
                or oracle.get('tile_count') != 8):
            raise ValueError('incomplete full-corpus K8 reconstruction/oracle report: ' + module)
        if 'identity' in row and row['identity'] != identity:
            raise ValueError('K8 diagnostic producer identity differs')
        records[module] = row
    if records.keys() != selection.keys():
        raise ValueError('K8 diagnostics do not cover exact reviewed module set')
    return records


def assemble(family, parent, source, output, *, k8=None, k8_evidence=None,
             parent_revision=None, shard_target=2_000_000_000):
    recipe, selection = family_recipe(family)
    parent, source, output = [Path(p).resolve() for p in (parent, source, output)]
    inputs = [parent, source] + ([Path(k8).resolve()] if k8 else [])
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError('package output must be disjoint from all immutable inputs')
    expected_parent = PARENTS.get(family, (None, parent_revision))[1]
    if source.name != recipe.BASE_REVISION or parent.name != expected_parent:
        raise ValueError('source/parent snapshot must use exact reviewed revisions')
    if family == 'kimi-k31' and (not parent_revision or not re.fullmatch('[0-9a-f]{40}', parent_revision)):
        raise ValueError('Kimi requires the completed, reviewed parent publication SHA')
    if family != 'ml4' and k8 is None:
        raise ValueError('MC package requires complete calibrated K8 export')
    parent_inv = full_parent_inventory(parent)
    source_index = 'consolidated.safetensors.index.json' if family == 'ml4' else 'model.safetensors.index.json'
    reviewed_source_bytes = (recipe.BASE_PAYLOAD_BYTES
                             if hasattr(recipe, 'inventory') and hasattr(recipe, 'BASE_PAYLOAD_BYTES')
                             else None)
    source_inv = inventory(source, source_index, reviewed_payload_bytes=reviewed_source_bytes)
    from types import SimpleNamespace
    document = load_json(source / 'config.json') if (source / 'config.json').exists() else None
    if hasattr(recipe, 'inventory'):
        adapter = SimpleNamespace(snapshot=source, weight_map=source_inv['mapping'], document=document,
            config=document if family in ('ml4', 'kimi-k31', 'glm5') else document['text_config'],
            text_config=document.get('text_config'))
        recipe.inventory(adapter)
    quant = load_json(parent / 'quantize_config.json')
    config = load_json(parent / 'config.json')
    removed = {name for module in selection for name in packed_specs(module, dict(selection[module], bits=6))}
    for module, geometry in selection.items():
        for name, spec in packed_specs(module, dict(geometry, bits=6)).items():
            if parent_inv['specs'].get(name) != spec:
                raise ValueError('parent K6 core does not match reviewed geometry: ' + name)
    native = restored_names(family, selection)
    if not native <= source_inv['mapping'].keys() or native & parent_inv['mapping'].keys():
        raise ValueError('native restore inventory missing or collides with shared parent')
    for module, geometry in selection.items():
        expected_dtype = 'F8_E4M3' if family == 'ml4' and len(recipe.source_names(module, geometry)) == 2 else 'BF16'
        if source_inv['specs'][module + '.weight'] != (expected_dtype, [geometry['out_features'], geometry['in_features']]):
            raise ValueError('native core source shape/dtype differs: ' + module)
    identity = dict(schema='cuteafd-core-package-v1', family=family,
        source_repo=recipe.BASE_REPO, source_revision=recipe.BASE_REVISION,
        parent_revision=expected_parent, tool_sha256=sha256_file(Path(__file__)),
        parent_index_sha256=sha256_file(parent / 'model.safetensors.index.json'),
        parent_config_sha256=sha256_file(parent / 'config.json'),
        parent_quantize_sha256=sha256_file(parent / 'quantize_config.json'),
        source_config_sha256=sha256_file(source / 'config.json') if (source / 'config.json').exists() else None,
        source_index_sha256=sha256_file(source / source_index),
        source_payload_bytes=dict(declared=source_inv['index'].get('metadata', {}).get('total_size'),
                                  measured=sum(source_inv['sizes'].values()),
                                  reviewed=reviewed_source_bytes),
        k8_receipt_sha256=sha256_file(Path(k8) / 'core-export.json') if k8 else None)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'package-plan.json'
    if plan_path.exists() and load_json(plan_path) != identity:
        raise ValueError('existing package requires identical immutable identity')
    atomic_json(plan_path, identity)
    new, copies, mapping = [], {}, {}
    receipts = {}
    # ML4 cannot contain inactive physical K6 core; repack every mixed shard.
    for shard, header in sorted(parent_inv['headers'].items()):
        names = set(header) - {'__metadata__'}
        kept = names - removed
        if not kept and family == 'ml4':
            continue
        if family == 'ml4' and names & removed:
            filename = 'native-parent-' + shard
            entries = {n: (parent / shard, n) for n in kept}
            receipts[filename] = write_shard(output, filename, entries, identity)
            new.append(filename)
        else:
            filename = shard
            path = output / filename
            if path.exists() or path.is_symlink():
                if not path.is_symlink() or path.resolve() != (parent / shard).resolve():
                    raise ValueError('inherited parent shard link differs')
            else:
                path.symlink_to(parent / shard)
            copies[filename] = shard
        mapping.update({n: filename for n in kept})
    native_map = {}
    for index, names in enumerate(shard_groups(native, source_inv['sizes'], shard_target), 1):
        filename = f'original-core-{index:04d}.safetensors'
        receipts[filename] = write_shard(output, filename,
            {n: (source / source_inv['mapping'][n], n) for n in names}, identity)
        new.append(filename)
        native_map.update({n: filename for n in names})
    native_specs = {n: source_inv['specs'][n] for n in native}
    views = {'original': (mapping | native_map, sorted(native), None)}
    if family != 'ml4':
        k8 = Path(k8)
        complete = load_json(k8 / 'core-export.json')
        if (complete['identity']['source_repo'] != recipe.BASE_REPO
                or complete['identity']['source_revision'] != recipe.BASE_REVISION
                or complete['modules'] != len(selection) or complete['bits'] != [8]):
            raise ValueError('K8 completion receipt source/modules/bitrate differs')
        if family in ('step5', 'glm5'):
            provenance = complete['identity']
            if not (provenance.get('gptqmodel_commit')
                    and provenance.get('encoder_delta', {}).get('k6_gptqmodel_commit')
                    and provenance.get('encoder_delta', {}).get('change')
                    and provenance.get('model_runtime', {}).get('transformers_version')
                    and provenance.get('packages', {}).get('torch')):
                raise ValueError('seed K8 receipt requires encoder delta and actual model runtime')
        k8_inv = full_parent_inventory(k8)
        expected = {n: spec for module, g in selection.items()
                    for n, spec in packed_specs(module, dict(g, bits=8)).items()}
        if k8_inv['specs'] != expected:
            raise ValueError('K8 inventory contains missing, native or expert tensors')
        diagnostics = k8_diagnostics(k8_evidence or k8 / 'quantization-errors', selection, complete['identity'])
        for module, row in sorted(diagnostics.items()):
            filename = 'quantization-errors.k8/' + module + '.json'
            atomic_json(output / filename, row)
            new.append(filename)
        atomic_json(output / 'quantization-provenance.k8.json', complete['identity'])
        new.append('quantization-provenance.k8.json')
        k8_map = {}
        for shard in k8_inv['headers']:
            producer_receipt = load_json(k8 / (shard + '.json'))
            if producer_receipt.get('identity') != complete['identity']:
                raise ValueError('K8 shard receipt identity differs from completion')
            verify_receipt(k8 / shard, producer_receipt, checksum=True)
            filename = 'k8-' + shard
            # Reuse bounded assembler rather than relying on cross-device hardlinks.
            names = set(k8_inv['headers'][shard]) - {'__metadata__'}
            receipts[filename] = write_shard(output, filename, {n: (k8 / shard, n) for n in names}, identity)
            new.append(filename)
            k8_map.update({n: filename for n in names})
        views['k8-compact'] = (mapping | k8_map, sorted(k8_map), 8)
        views['k6-compact'] = (dict(parent_inv['mapping']), sorted(removed), 6)
    # Keep tokenizer, license, bundled drafter assets and parent diagnostics.
    # Parent cards/provenance stay explicitly historical, not claims about MC.
    assets = []
    for path in sorted(parent.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(parent).as_posix()
        if (relative in parent_inv['headers']
                or ('/' not in relative and relative.endswith('.safetensors.index.json'))
                or relative in ('config.json', 'quantize_config.json', '.gitattributes')
                or path.name.endswith('.receipt.json')):
            continue
        target = 'README.parent.md' if relative == 'README.md' else relative
        assets.append((target, relative))
    for target, relative in assets:
        path = output / target
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != (parent / relative).resolve():
                raise ValueError('inherited asset link differs')
        else:
            path.symlink_to(parent / relative)
        copies[target] = relative
    metadata = []
    for view, (weights, core, bits) in views.items():
        index_file, config_file, quant_file, _ = mc_manifest.FILES[view]
        q, c = view_metadata(quant, config, selection, native_specs,
                            bits=bits, quantize_file=quant_file)
        sizes = parent_inv['sizes'] | {n: source_inv['sizes'][n] for n in native}
        if bits == 8:
            sizes.update(k8_inv['sizes'])
        atomic_json(output / index_file, dict(metadata=dict(total_size=sum(sizes[n] for n in weights)),
                                             weight_map=dict(sorted(weights.items()))))
        atomic_json(output / config_file, c)
        atomic_json(output / quant_file, q)
        metadata += [index_file, config_file, quant_file]
    if family != 'ml4':
        manifest = mc_manifest.build(output, source_repo=recipe.BASE_REPO, source_revision=recipe.BASE_REVISION,
                                     core_tensors={view: row[1] for view, row in views.items()})
        atomic_json(output / mc_manifest.MANIFEST, manifest)
        metadata.append(mc_manifest.MANIFEST)
    elif (parent / 'consolidated.safetensors.index.json').exists():
        atomic_json(output / 'consolidated.safetensors.index.json', load_json(output / 'model.safetensors.index.json'))
        metadata.append('consolidated.safetensors.index.json')
    parent_card = (parent / 'README.md').read_text() if (parent / 'README.md').exists() else ''
    front_matter = ''
    if parent_card.startswith('---\n'):
        boundary = parent_card.find('\n---\n', 4)
        if boundary < 0:
            raise ValueError('parent model-card front matter is unterminated')
        front_matter = parent_card[:boundary + 5] + '\n'
    card = (front_matter + '# ' + recipe.BASE_REPO.split('/')[-1] + (' EXL3 K4 native core\n\n' if family == 'ml4' else ' EXL3 multi-core\n\n')
        + 'Derived from `' + recipe.BASE_REPO + '@' + recipe.BASE_REVISION + '`.\n\n'
        + ('This plain checkpoint keeps the original FP8/BF16 core and K4 experts. No quantized core payload is included.\n\n'
           if family == 'ml4' else 'The default index selects the original BF16 core. `cuteafd-mc.json` discovers complete K8 and K6 alternate views; experts, media and drafters are shared.\n\n')
        + ('Only target-core matrices change between views. The parent K4 MTP dense modules '
           f'({27 if family == "step5" else 8} modules) remain shared unchanged across all three views; '
           'they are neither restored to BF16 nor encoded at K8.\n\n' if family in ('step5', 'glm5') else '')
        + (f"K8 encoder fork: `{complete['identity']['gptqmodel_commit']}`; "
           f"inherited K6 encoder fork: `{complete['identity']['encoder_delta']['k6_gptqmodel_commit']}`. "
           f"Delta: {complete['identity']['encoder_delta']['change']}. "
           f"K8 model runtime: Transformers `{complete['identity']['model_runtime']['transformers_version']}`, "
           f"Torch `{complete['identity']['packages']['torch']}`. "
           'The K8 encoder build need not match the historical K6 build. Full source hashes, '
           'calibration, recipe and actual dependency versions are recorded in '
           '`quantization-provenance.k8.json`.\n\n' if family in ('step5', 'glm5') else '')
        + 'Original-core byte proofs and package inventory are in `package-validation.json`. Parent diagnostics describe the parent quantization, not the restored core. '
        + 'See `README.parent.md` and inherited license/usage files for checkpoint terms. No generation quality or serving performance qualification is claimed by this assembly.\n')
    (output / 'README.md').write_text(card)
    (output / '.gitattributes').write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    metadata += ['README.md', '.gitattributes']
    report = dict(identity=identity, new_files=sorted(new + metadata), copies=copies,
        new_shards=receipts,
        core_tensors={view: row[1] for view, row in views.items()},
        source_tensor_locations={n: source_inv['mapping'][n] for n in native},
        source_snapshot=str(source), parent_snapshot=str(parent),
        k8_calibration_count_key=calibration_count_key(complete['identity']) if k8 else None)
    atomic_json(output / 'package-state.json', report)
    atomic_json(output / 'package-provenance.json', identity)
    return validate(output)


def audit_ml4(output):
    """Independently prove every retained tensor against pinned parent bytes."""
    output = Path(output)
    state = load_json(output / 'package-state.json')
    if state['identity']['family'] != 'ml4':
        raise ValueError('ML4 physical byte audit requires an ML4 package')
    parent = full_parent_inventory(state['parent_snapshot'])
    target = full_parent_inventory(output)
    recipe, selection = family_recipe('ml4')
    removed = {n for module, geometry in selection.items()
               for n in packed_specs(module, dict(geometry, bits=6))}
    native = restored_names('ml4', selection)
    shared = parent['mapping'].keys() - removed
    if target['mapping'].keys() != shared | native:
        raise ValueError('ML4 exact target tensor inventory differs')
    for path in output.rglob('*.safetensors'):
        header, _ = read_header(path)
        if removed.intersection(header):
            raise ValueError('ML4 physically contains quantized core: ' + str(path))
    proofs = {}
    for shard in sorted(parent['headers']):
        names = sorted(n for n in shared if parent['mapping'][n] == shard)
        with ExitStack() as stack:
            origin = Path(state['parent_snapshot']) / shard
            source_header, source_base = read_header(origin)
            source_stream = stack.enter_context(origin.open('rb'))
            targets = {}
            for name in names:
                destination = output / target['mapping'][name]
                if destination not in targets:
                    header, base = read_header(destination)
                    targets[destination] = (stack.enter_context(destination.open('rb')), header, base)
                stream, header, base = targets[destination]
                start, end = source_header[name]['data_offsets']
                left, right = header[name]['data_offsets']
                if ((source_header[name]['dtype'], source_header[name]['shape'], end-start)
                        != (header[name]['dtype'], header[name]['shape'], right-left)):
                    raise ValueError('ML4 inherited tensor geometry differs: ' + name)
                source_stream.seek(source_base + start)
                stream.seek(base + left)
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    count = min(16 << 20, remaining)
                    source_bytes, target_bytes = source_stream.read(count), stream.read(count)
                    if len(source_bytes) != count or source_bytes != target_bytes:
                        raise ValueError('ML4 inherited tensor bytes differ: ' + name)
                    digest.update(source_bytes)
                    remaining -= count
                proofs[name] = digest.hexdigest()
    native_proofs = {}
    source = inventory(state['source_snapshot'], 'consolidated.safetensors.index.json')
    for name in sorted(native):
        digest = tensor_hash(source['root'] / source['mapping'][name], name)
        if tensor_hash(output / target['mapping'][name], name) != digest:
            raise ValueError('ML4 original core differs: ' + name)
        native_proofs[name] = digest
    receipt = dict(schema='cuteafd-ml4-independent-byte-audit-v1', identity=state['identity'],
        status='all-native-and-shared-bytes-exact-no-physical-quantized-core',
        inherited_tensor_sha256=proofs, original_core_tensor_sha256=native_proofs,
        inherited_tensors=len(proofs), inherited_payload_bytes=sum(parent['sizes'][n] for n in shared),
        core_tensors=len(native_proofs), core_payload_bytes=sum(source['sizes'][n] for n in native),
        physical_safetensors=len(list(output.rglob('*.safetensors'))))
    atomic_json(output.parent / 'independent-byte-audit.json', receipt)
    return {k:v for k,v in receipt.items() if not k.endswith('tensor_sha256')}


def validate(output):
    output = Path(output)
    state = load_json(output / 'package-state.json')
    if state['identity']['schema'] != 'cuteafd-core-package-v1':
        raise ValueError('unsupported package receipt')
    for filename, receipt in state['new_shards'].items():
        verify_receipt(output / filename, receipt, checksum=True)
    family = state['identity']['family']
    if family != 'ml4':
        mc_manifest.validate(load_json(output / mc_manifest.MANIFEST), output)
    views = {}
    for view, core in state['core_tensors'].items():
        index, config, quantize, _ = mc_manifest.FILES[view]
        inv = inventory(output, index)
        q = load_json(output / quantize)
        declared = {}
        for module, entry in q['tensor_storage'].items():
            for name, tensor in entry['stored_tensors'].items():
                dtype = next(k for k, v in DTYPES.items() if v == tensor['torch_dtype'])
                declared[name] = (dtype, tensor['shape'])
        if any(inv['specs'].get(n) != spec for n, spec in declared.items()):
            raise ValueError('selected quant metadata differs from tensor headers')
        for name, tensor in q.get('retained_native_tensors', {}).items():
            if (set(tensor) != {'dtype', 'shape'}
                    or inv['specs'].get(name) != (tensor['dtype'], tensor['shape'])):
                raise ValueError('retained native metadata differs from tensor headers')
        if family == 'ml4':
            _, selected = family_recipe(family)
            if set(selected) & q['tensor_storage'].keys():
                raise ValueError('ML4 still declares quantized core')
            forbidden = {m + '.' + suffix for m in selected for suffix in SUFFIXES}
            if any(forbidden & set(header) for header in inv['headers'].values()):
                raise ValueError('ML4 physically contains inactive quantized core')
        views[view] = dict(tensors=len(inv['mapping']), payload_bytes=sum(inv['sizes'].values()),
                           core_tensors=len(core))
    original = inventory(output)
    original_native_specs = original['specs']
    _, reviewed_selection = family_recipe(family)
    for module, geometry in reviewed_selection.items():
        if family == 'ml4':
            recipe, _ = family_recipe(family)
            names = recipe.source_names(module, geometry)
            if len(names) == 2 and original_native_specs.get(names[1]) != ('F32', [
                    (geometry['out_features'] + 127) // 128, geometry['in_features'] // 128]):
                raise ValueError('restored FP8 scales missing or malformed')
    proofs = {}
    source = Path(state['source_snapshot'])
    for name, shard in sorted(state['source_tensor_locations'].items()):
        digest = tensor_hash(source / shard, name)
        if tensor_hash(output / original['mapping'][name], name) != digest:
            raise ValueError('restored original core payload differs: ' + name)
        proofs[name] = digest
    receipt = dict(schema='cuteafd-package-validation-v1', identity=state['identity'], views=views,
                   original_core_tensor_sha256=proofs,
                   new_file_bytes=sum((output / n).stat().st_size for n in state['new_files']),
                   new_file_sha256={n: sha256_file(output / n) for n in state['new_files']},
                   copied_file_sha256={n: sha256_file(output / n) for n in state['copies'] if not n.endswith('.safetensors')},
                   copied_files=len(state['copies']), status='validated-headers-and-byte-exactness',
                   k8_calibration_count_key=state.get('k8_calibration_count_key'))
    atomic_json(output / 'package-validation.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['kimi-k31', 'glmf', 'muse', 'ml4', 'step5', 'glm5'])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parent-revision')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--k8', type=Path)
    parser.add_argument('--k8-evidence', type=Path,
                        help='Complete producer evidence tree; default K8 export/quantization-errors')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--audit-ml4', action='store_true',
                        help='Independently compare every inherited/native tensor payload')
    args = parser.parse_args()
    if args.audit_ml4:
        result = audit_ml4(args.output)
    elif args.validate_only:
        result = validate(args.output)
    else:
        if not args.family or not args.parent or not args.source:
            parser.error('build requires family, parent, source and output')
        result = assemble(args.family, args.parent, args.source, args.output,
                          k8=args.k8, k8_evidence=args.k8_evidence, parent_revision=args.parent_revision)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
