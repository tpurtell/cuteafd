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
import os
from pathlib import Path
import re
import sys
import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/lib'))
import mc_manifest
from core_export import CODEBOOK_MULTIPLIERS, MCG, MUL1, SUFFIXES, packed_specs
from kimi_k3.uniform_k2.storage import (
    assemble_shard, atomic_json, canonical, file_identity, fsync_directory, read_header, sha256_file, verify_receipt,
)

DTYPES = {'BF16': 'bfloat16', 'F32': 'float32', 'F16': 'float16',
          'F8_E4M3': 'float8_e4m3fn', 'I16': 'int16', 'I32': 'int32'}
# Families that share the strict direct-writer MUL1 contract.
DIRECT_MUL1 = {"hy4_flash", "hy4", "ernie55_flash"}


def check_source_shard(family, shard):
    # ERNIE's twelve additional source tensors live outside the main index.
    if family == "ernie55_flash" and shard == "flashindexer/flashindexer.safetensors":
        return
    if not shard or shard in (".", "..") or Path(shard).name != shard or any(c in shard for c in "\\\0\n\r"):
        raise ValueError("unsafe original source shard")


def native_direct_selection(family, compact, config):
    """Resolve the native row layout corresponding to the compact module inventory."""
    if family == "ernie55_flash":
        from ernie55_flash.hca_format import original_selection
        return original_selection(compact, config)
    return compact


def original_tensor_name(family, module):
    if family == "ernie55_flash" and module.endswith(".self_attn.linear_o_group_proj"):
        return module
    return module + ".weight"


PARENTS = {
    'glmf': ('GLM-5.3-Flash-EXL3-K4-Core-K6', '5d3fffe1df9c4a109bbbdafb60dc19df9fac019d'),
    'glmf-k325': ('GLM-5.3-Flash-EXL3-K4-MC', '91a8dbc9c12e790c65aa1a38c6aede6a3e23a77d'),
    'muse': ('Muse-Spark-Open-EXL3-K4-Core-K6', '89998fb177acd9a862ac8bd4fb79bfd50e940c97'),
    'ml4': ('Mistral-Large-4-EXL3-K4-Core-K6', '03c80aebaa483d55760703ae5bc0c9734c9d67d6'),
    'step5': ('Step-5-Preview-EXL3-K4-Core-K6', '83d47a384eabc07ea6fcfaac4779b0e5f013ce89'),
    'glm5': ('GLM-5.3-EXL3-K4-Core-K6', '5a4e83a09a11f3145332ed632d28a9894ea88225'),
}


def tensor_storage_entry(geometry, stored_tensors, *, mcg_multiplier=MCG,
                         mul1_multiplier=None, **extra):
    """Build the common EXL3 record without discarding family-specific fields."""
    entry = dict(quant_format='exl3', bits_per_weight=geometry['bits'],
                 **{key: geometry[key] for key in
                    ('in_features', 'out_features', 'stored_out_features')},
                 stored_tensors=stored_tensors, **extra)
    if mcg_multiplier is not None:
        entry['mcg_multiplier'] = mcg_multiplier
    if mul1_multiplier is not None:
        entry['mul1_multiplier'] = mul1_multiplier
    return entry


def quantize_config_for_view(quantization_config, **metadata):
    """Mirror the view's inline declaration into its per-module metadata file."""
    return dict(metadata, **deepcopy(quantization_config))


def gptqmodel_mc_view(config, metadata, *, bits, core_bits=None, module_include, codebook="mcg"):
    """Package MC storage using GPTQModel's declaration and provenance schema."""
    config, metadata = deepcopy(config), deepcopy(metadata)
    previous = config['quantization_config']
    declaration = dict(quant_method='exl3', format='exl3', checkpoint_format='exl3',
                       bits=bits, per_module_metadata=previous['per_module_metadata'])
    if codebook not in CODEBOOK_MULTIPLIERS:
        raise ValueError('unknown GPTQModel MC codebook')
    if codebook != 'mcg':
        declaration.update(codebook=codebook, codebook_multiplier=CODEBOOK_MULTIPLIERS[codebook])
    if core_bits is not None:
        declaration['core_bits'] = core_bits
    config['quantization_config'] = declaration
    meta = metadata.setdefault('meta', {})
    meta['source_quantization_config'] = previous
    for key in ('provenance', 'core_view', 'upstream_activations', 'activation_quantization'):
        if key in metadata:
            meta[key] = metadata.pop(key)
    metadata.pop('format', None)
    storage = metadata['tensor_storage']
    metadata.update(method='exl3', codebook=codebook, out_scales='auto', group_size=-1,
                    desc_act=False, module_include=list(module_include),
                    lm_head=any(name == 'lm_head' or name.endswith('.lm_head') or name == 'output'
                                for name in storage))
    return quantize_config_for_view(declaration, **metadata), config


def family_recipe(family):
    """Return the completed numerical selection used by compact encodes."""
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


def original_core_selection(family, recipe, compact_selection):
    """Restore the whole native core, independently of compact calibration.

    Historical GLM and Step compact encodes selected only the target core;
    their MTP matrices already exist at K4 in the parent. Those matrices must
    also be restored in the original view without requiring new K6/K8 work.
    """
    if family == 'glmf':
        return recipe.reviewed_core_modules(include_mtp=True, core_bits=8)
    if family in ('glm5', 'step5'):
        return recipe.reviewed_core_modules(include_mtp=True)
    return compact_selection


def load_json(path):
    return json.loads(Path(path).read_text())


def inventory(root, index_name='model.safetensors.index.json', *, reviewed_payload_bytes=None):
    root = Path(root)
    index = load_json(root / index_name)
    mapping = index['weight_map']
    headers, bases, specs, sizes = {}, {}, {}, {}
    for shard in sorted(set(mapping.values())):
        if Path(shard).name != shard or not shard.endswith('.safetensors'):
            raise ValueError('index shard must be a safetensors basename')
        headers[shard], bases[shard] = read_header(root / shard)
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
    return dict(root=root, index=index, mapping=mapping, headers=headers, bases=bases, specs=specs, sizes=sizes)


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


def _assemble_proved_shard(entries, path, *, chunk_bytes=16 << 20):
    """Hash original tensor spans during their single source-to-output copy.

    Each digest covers the exact byte objects written into the final output;
    this is a streaming copy proof, not a later payload reread. Source stat
    identities and aliases bind it to the reviewed immutable source inventory.
    """
    path = Path(path)
    if path.exists() or not entries or not 1 <= chunk_bytes <= 64 << 20:
        raise ValueError('invalid proved shard output or chunk bound')
    sources, header, spans, offset = {}, {}, [], 0
    for name, (source_path, source_name) in sorted(entries.items()):
        source_path = Path(source_path)
        if source_path not in sources:
            sources[source_path] = read_header(source_path)
        source_header, base = sources[source_path]
        value = source_header[source_name]
        start, end = value['data_offsets']
        header[name] = dict(dtype=value['dtype'], shape=value['shape'],
                            data_offsets=[offset, offset + end - start])
        offset += end - start
        spans.append((name, source_path, source_name, base + start, end - start))
    encoded = canonical(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    if temporary.is_symlink():
        raise ValueError('refuse proved shard temporary symlink')
    digest, proofs = hashlib.sha256(), {}
    with temporary.open('wb') as output:
        prefix = struct.pack('<Q', len(encoded)) + encoded
        output.write(prefix)
        digest.update(prefix)
        for name, source_path, source_name, position, remaining in spans:
            before = file_identity(source_path)
            tensor_digest = hashlib.sha256()
            with source_path.open('rb') as source:
                source.seek(position)
                while remaining:
                    block = source.read(min(remaining, chunk_bytes))
                    if not block:
                        raise ValueError('truncated original source tensor')
                    output.write(block)
                    digest.update(block)
                    tensor_digest.update(block)
                    remaining -= len(block)
            if file_identity(source_path) != before:
                raise ValueError('original source changed during streaming copy')
            proofs[name] = dict(sha256=tensor_digest.hexdigest(), source_name=source_name,
                                source_path=str(source_path.resolve()), source_file_identity=before)
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != 8 + len(encoded) + offset:
        raise ValueError('proved shard length differs')
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return dict(sha256=digest.hexdigest(), file_identity=file_identity(path), payload_bytes=offset,
                original_tensor_proofs=proofs)


def write_shard(root, filename, entries, identity, *, prove_original=False, verify_checksum=True):
    path = root / filename
    receipt_path = root / (filename + '.receipt.json')
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if receipt['identity'] != identity or receipt['tensors'] != sorted(entries):
            raise ValueError('package recovery identity differs')
        verify_receipt(path, receipt, checksum=verify_checksum)
        if prove_original and set(receipt.get('original_tensor_proofs', {})) != set(entries):
            raise ValueError('original streaming proofs missing from committed shard')
    else:
        if path.is_symlink():
            raise ValueError('refuse package output symlink')
        path.unlink(missing_ok=True)
        assembler = _assemble_proved_shard if prove_original else assemble_shard
        receipt = dict(assembler(entries, path), identity=identity, tensors=sorted(entries))
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
            entry.update(tensor_storage_entry(dict(geometry, bits=bits),
                         {name: dict(shape=shape, torch_dtype=DTYPES[dtype])
                          for name, (dtype, shape) in packed_specs(module, dict(geometry, bits=bits)).items()}))
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
        quant.pop('mtp_bits', None)
        if 'kept_tensors' in quant:
            quant['kept_tensors'].update({name: dict(shape=shape, torch_dtype=DTYPES[dtype])
                for name, (dtype, shape) in specs.items()})
        if 'retained_native_tensors' in quant:
            quant['retained_native_tensors'].update({name: dict(shape=shape, dtype=dtype)
                for name, (dtype, shape) in specs.items()})
    quant['per_module_metadata'] = quantize_file + ':tensor_storage'
    def rewrite_inline(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'quantization_config':
                    # Multimodal text configs carry their own copy of EXL3 metadata.
                    child['per_module_metadata'] = quant['per_module_metadata']
                    for field in ('core_bits', 'head_bits'):
                        if bits is None:
                            child.pop(field, None)
                        else:
                            child[field] = bits
                    if bits is None:
                        child.pop('mtp_bits', None)
                rewrite_inline(child)
        elif isinstance(value, list):
            for child in value:
                rewrite_inline(child)

    rewrite_inline(config)
    quant = quantize_config_for_view(config.get('quantization_config', {}), **quant)
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
    original_selection = original_core_selection(family, recipe, selection)
    if not selection.keys() <= original_selection.keys():
        raise ValueError('original core selection omits compact core modules')
    inherited_core = {m: g for m, g in original_selection.items() if m not in selection}
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
    compact_removed = {name for module in selection for name in packed_specs(module, dict(selection[module], bits=6))}
    inherited_packed = {name: spec for module, geometry in inherited_core.items()
                        for name, spec in packed_specs(module, geometry).items()}
    removed = compact_removed | inherited_packed.keys()
    for module, geometry in selection.items():
        for name, spec in packed_specs(module, dict(geometry, bits=6)).items():
            if parent_inv['specs'].get(name) != spec:
                raise ValueError('parent K6 core does not match reviewed geometry: ' + name)
    for name, spec in inherited_packed.items():
        if parent_inv['specs'].get(name) != spec:
            raise ValueError('parent MTP core does not match reviewed geometry: ' + name)
    native = restored_names(family, original_selection)
    if not native <= source_inv['mapping'].keys() or native & parent_inv['mapping'].keys():
        raise ValueError('native restore inventory missing or collides with shared parent')
    for module, geometry in original_selection.items():
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
        inherited_map = {name: parent_inv['mapping'][name] for name in inherited_packed}
        views['k8-compact'] = (mapping | inherited_map | k8_map, sorted(k8_map.keys() | inherited_packed.keys()), 8)
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
        metadata_selection = original_selection if bits is None else selection
        q, c = view_metadata(quant, config, metadata_selection, native_specs,
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
           if family == 'ml4' else 'The default index selects the original BF16 core. `cuteafd-mc.json` discovers complete K8 and K6 alternate views; routed experts and unchanged native tensors are shared.\n\n')
        + ('The original view restores the complete source core, including '
           f'{len(inherited_core)} MTP matrices. Compact views preserve the existing K4 MTP '
           'encodes; their K8/K6 labels describe the target core. Routed experts remain shared.\n\n'
           if inherited_core else '')
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


def _validate_direct_engram(output, manifest, state):
    """Bind unindexed alternate headers to all-row and streamed write receipts."""
    engram = manifest.get('engram_fp8')
    proof = state.get('engram_fp8_proof', {})
    if state['identity']['family'] != 'hy4_flash':
        if engram is not None and 'direct_expected_specs' in state:
            raise ValueError('unexpected Engram alternate in direct package')
        return
    from hy4_flash import recipe as hy4
    if engram is None or proof.keys() != hy4.ENGRAM_TABLES.keys():
        raise ValueError('Hy4-flash requires all four proven Engram alternates')
    if {layer['tensor'] for layer in engram['layers']} != hy4.ENGRAM_TABLES.keys():
        raise ValueError('Engram alternate tensor set differs from the pinned sources')
    if not engram['recipe'].endswith('; reference=' + hy4.REFERENCE_REVISION):
        raise ValueError('Engram alternate reference revision differs')
    table_files = set()
    for layer in engram['layers']:
        tensor = layer['tensor']
        expected_rows = hy4.ENGRAM_TABLES[tensor]
        verification = proof[tensor]
        if (verification.get('rows') != expected_rows or verification.get('mismatch_count') != 0
                or verification.get('reference_dtype') != 'BF16'
                or verification.get('expression') != 'x + (decoded - x)'):
            raise ValueError('Engram all-row proof differs')
        native = state['direct_expected_mappings']['original'][tensor]
        header, _ = read_header(output / native)
        if set(header) - {'__metadata__'} != {tensor}:
            raise ValueError('BF16 Engram shard must contain only its table')
        if (state['new_shards'][native]['original_tensor_proofs'][tensor]['sha256']
                != layer['sha256']):
            raise ValueError('BF16 Engram manifest tensor SHA differs from writer proof')
        offset = 0
        for part in layer['parts']:
            filename = part['file']
            table_files.add(filename)
            record = state['new_shards'][filename]
            header, _ = read_header(output / filename)
            metadata = header.get('__metadata__', {})
            spec = header.get(part['tensor'], {})
            if (record['sha256'] != part['sha256'] or set(header) - {'__metadata__'} != {part['tensor']}
                    or spec.get('dtype') != 'U8' or spec.get('shape') != [part['rows'], engram['row_bytes']]
                    or metadata.get('engram_layout') != engram['layout']
                    or metadata.get('row_start') != str(offset)):
                raise ValueError('FP8 Engram header or writer receipt differs')
            offset += part['rows']
        if offset != expected_rows:
            raise ValueError('FP8 Engram parts do not cover all source rows')
    selected_files = {filename for mapping in state['direct_expected_mappings'].values()
                      for filename in mapping.values()}
    if selected_files & table_files or set(state['new_shards']) != selected_files | table_files:
        raise ValueError('Hy4-flash alternate must be physically separate and only in the MC manifest')


def _check_direct_markers(output, inv, markers, checked):
    """Read four bytes per physical scalar, sharing shard handles across modules."""
    by_shard = {}
    for name, codebook in markers.items():
        filename = inv['mapping'][name]
        if (filename, name) not in checked:
            offset = inv['headers'][filename][name]['data_offsets'][0]
            by_shard.setdefault(filename, []).append((offset, name, codebook))
    for filename, extents in by_shard.items():
        with (output / filename).open('rb', buffering=0) as stream:
            for offset, name, codebook in sorted(extents):
                stream.seek(inv['bases'][filename] + offset)
                if stream.read(4) != struct.pack('<I', CODEBOOK_MULTIPLIERS[codebook]):
                    raise ValueError(f'direct {codebook.upper()} multiplier differs: {name}')
                checked.add((filename, name))


def _direct_mul1_markers(inv, quantize, config, core, view):
    """Require full metadata for every indexed shared expert and compact projection."""
    if any(value.get('codebook') != 'mul1' or value.get('codebook_multiplier') != MUL1
           for value in (quantize, config['quantization_config'])):
        raise ValueError('Hy4 view differs from frozen MUL1 codebook')
    packed = {name.rsplit('.', 1)[0] for name in inv['mapping'] if name.endswith('.trellis')}
    if packed != quantize['tensor_storage'].keys():
        raise ValueError('Hy4 tensor_storage does not cover every selected EXL3 module')
    markers = {}
    for module, entry in quantize['tensor_storage'].items():
        width = entry.get('bits_per_weight')
        if module not in core and width != 4:
            raise ValueError('Hy4 routed expert requires K4')
        if (entry.get('quant_format') != 'exl3' or entry.get('side') != 1
                or entry.get('mul1_multiplier') != MUL1 or 'mcg_multiplier' in entry):
            raise ValueError('Hy4 EXL3 module codebook or side differs')
        geometry = dict(entry, bits=width, codebook='mul1')
        specs = packed_specs(module, geometry)
        if set(specs) != set(entry['stored_tensors']) or any(inv['specs'].get(n) != spec for n, spec in specs.items()):
            raise ValueError('Hy4 packed tensor geometry differs: ' + module)
        markers[module + '.mul1'] = 'mul1'
    native = set(inv['mapping']) - {n for e in quantize['tensor_storage'].values() for n in e['stored_tensors']}
    if native != quantize.get('retained_native_tensors', {}).keys():
        raise ValueError(f'Hy4 retained-native inventory differs in {view}')
    return markers


def validate(output, *, verification='full'):
    """Validate payloads by default; direct writers may reuse streaming proofs."""
    if verification not in ('full', 'streaming-receipts'):
        raise ValueError('unsupported package verification mode')
    output = Path(output)
    state = load_json(output / 'package-state.json')
    if state['identity']['schema'] != 'cuteafd-core-package-v1':
        raise ValueError('unsupported package receipt')
    family = state['identity']['family']
    if verification == 'streaming-receipts' and (family != 'inkling2' and family not in DIRECT_MUL1
            or 'direct_expected_specs' not in state or state.get('copies') != {}):
        raise ValueError('streaming receipts require an independent direct package')
    if verification == 'streaming-receipts':
        source_names = set(state['core_tensors']['original'])
        if family in DIRECT_MUL1:
            expert_prefix = '.block_sparse_moe.experts.' if family == 'ernie55_flash' else '.mlp.experts.'
            source_names.update(n for n in state['direct_expected_specs']['original']
                                if expert_prefix not in n)
        if set(state['source_tensor_locations']) != source_names:
            raise ValueError('original streaming proof inventory is incomplete')
    for filename, receipt in state['new_shards'].items():
        verify_receipt(output / filename, receipt, checksum=verification == 'full')
    if 'direct_expected_specs' in state:
        if set(p.name for p in output.glob('*.safetensors')) != set(state['new_shards']):
            raise ValueError('unowned physical weights in direct package')
        payload = sum(sum(v['data_offsets'][1] - v['data_offsets'][0]
                          for n, v in read_header(output / filename)[0].items() if n != '__metadata__')
                      for filename in state['new_shards'])
        if payload != state['direct_payload_bytes']:
            raise ValueError('direct physical payload differs from committed inventory')
    if family != 'ml4':
        manifest = mc_manifest.validate(load_json(output / mc_manifest.MANIFEST), output)
        _validate_direct_engram(output, manifest, state)
    if family in DIRECT_MUL1 and (state['identity'].get('codebook') != 'mul1'
            or state['identity'].get('codebook_multiplier') != MUL1
            or manifest.get('exl3_codebook') != dict(name='mul1', multiplier=MUL1)):
        raise ValueError('Hy4 package, manifest and producer must agree on MUL1')
    views = {}
    checked_markers = set()
    for view, core in state['core_tensors'].items():
        index, config, quantize, _ = mc_manifest.FILES[view]
        inv = inventory(output, index)
        q = load_json(output / quantize)
        if 'direct_expected_specs' in state:
            expected = {n: (s[0], s[1]) for n, s in state['direct_expected_specs'][view].items()}
            if inv['specs'] != expected or inv['mapping'] != state['direct_expected_mappings'][view]:
                raise ValueError('direct selected index differs from committed tensor inventory')
            selected = state['direct_core_selection']
            markers = (_direct_mul1_markers(inv, q, load_json(output / config), selected, view)
                       if family in DIRECT_MUL1 else {})
            for module, geometry in selected.items():
                if view == 'original':
                    if module in q['tensor_storage']:
                        raise ValueError('original direct view still declares quantized core')
                    continue
                bits = mc_manifest.FILES[view][3]
                if view == 'k6-compact' and module in state.get('direct_k6_bit_overrides', {}):
                    if selected[module].get('group') != 'mtp' or state['direct_k6_bit_overrides'][module] != 8:
                        raise ValueError('invalid K6 drafter fallback')
                    bits = 8
                    original_map = state['direct_expected_mappings']['k8-compact']
                    if any(inv['mapping'].get(name) != original_map[name]
                           for name in packed_specs(module, dict(geometry, bits=8))):
                        raise ValueError('K6 drafter fallback must alias the K8 shard')
                entry = q['tensor_storage'].get(module, {})
                if entry.get('bits_per_weight') != bits:
                    raise ValueError('direct compact core metadata width differs')
                required = packed_specs(module, dict(geometry, bits=bits))
                if set(entry.get('stored_tensors', {})) != set(required):
                    raise ValueError('direct compact core metadata inventory differs')
                codebook = geometry.get('codebook', 'mcg')
                markers[module + '.' + codebook] = codebook
            _check_direct_markers(output, inv, markers, checked_markers)
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
    reviewed_selection = state.get('direct_core_selection')
    if reviewed_selection is None:
        recipe, compact_selection = family_recipe(family)
        reviewed_selection = original_core_selection(family, recipe, compact_selection)
    reviewed_selection = native_direct_selection(family, reviewed_selection, load_json(output / 'config.json'))
    original_quant = load_json(output / 'quantize_config.json')
    for module, geometry in reviewed_selection.items():
        if module in original_quant['tensor_storage']:
            raise ValueError('original view still declares quantized core: ' + module)
        if family != 'ml4' and original_native_specs.get(original_tensor_name(family, module)) != (
                'BF16', [geometry['out_features'], geometry['in_features']]):
            raise ValueError('original view omits native core or differs from source geometry: ' + module)
        if family == 'ml4':
            recipe, _ = family_recipe(family)
            names = recipe.source_names(module, geometry)
            if len(names) == 2 and original_native_specs.get(names[1]) != ('F32', [
                    (geometry['out_features'] + 127) // 128, geometry['in_features'] // 128]):
                raise ValueError('restored FP8 scales missing or malformed')
    proofs = {}
    source = Path(state['source_snapshot'])
    for name, location in sorted(state['source_tensor_locations'].items()):
        # Direct sources can carry unindexed, bare-name MTP tensors.
        shard = location['shard'] if isinstance(location, dict) else location
        source_name = location['source_name'] if isinstance(location, dict) else name
        check_source_shard(family, shard)
        if verification == 'streaming-receipts':
            receipt = state['new_shards'][original['mapping'][name]]
            proof = receipt.get('original_tensor_proofs', {}).get(name, {})
            if (proof.get('source_path') != str((source / shard).resolve())
                    or proof.get('source_name') != source_name
                    or not re.fullmatch('[0-9a-f]{64}', proof.get('sha256', ''))):
                raise ValueError('original streaming copy proof or source alias differs: ' + name)
            if file_identity(source / shard) != proof.get('source_file_identity'):
                raise ValueError('original source file changed: ' + name)
            digest = proof['sha256']
        else:
            digest = tensor_hash(source / shard, source_name)
            if tensor_hash(output / original['mapping'][name], name) != digest:
                raise ValueError('restored original core payload differs: ' + name)
        proofs[name] = digest
    receipt = dict(schema='cuteafd-package-validation-v1', identity=state['identity'], views=views,
                   original_core_tensor_sha256=proofs,
                   new_file_bytes=sum((output / n).stat().st_size for n in state['new_files']),
                   new_file_sha256={n: (state['new_shards'][n]['sha256']
                       if verification == 'streaming-receipts' and n in state['new_shards']
                       else sha256_file(output / n)) for n in state['new_files']},
                   copied_file_sha256={n: sha256_file(output / n) for n in state['copies'] if not n.endswith('.safetensors')},
                   copied_files=len(state['copies']), status='validated-headers-and-byte-exactness',
                   k8_calibration_count_key=state.get('k8_calibration_count_key'))
    if verification == 'streaming-receipts':
        receipt['verification_mode'] = 'streaming-write-proofs-stat-headers-and-source-aliases'
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
    parser.add_argument('--verification', choices=('full', 'streaming-receipts'), default='full')
    parser.add_argument('--audit-ml4', action='store_true',
                        help='Independently compare every inherited/native tensor payload')
    args = parser.parse_args()
    if args.audit_ml4:
        result = audit_ml4(args.output)
    elif args.validate_only:
        result = validate(args.output, verification=args.verification)
    else:
        if not args.family or not args.parent or not args.source:
            parser.error('build requires family, parent, source and output')
        result = assemble(args.family, args.parent, args.source, args.output,
                          k8=args.k8, k8_evidence=args.k8_evidence, parent_revision=args.parent_revision)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
