#!/usr/bin/env python3
"""Bounded source-causal Mistral projection pilot, with exact native witnesses.

Two complete calibration and two heldout prompts traverse all 72 *source*
layers. No candidate ever modifies that trajectory. A separate synthetic
2205-token witness tests sparse selection and exact KV/indexer reuse; it is
correctness evidence, never a claim about long-context generation quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'quantization'))
if __package__ in (None, ''):
    __package__ = 'mistral_large4.core_k6'
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

from kimi_k3.uniform_k2.storage import (RollingBoundary, atomic_json, canonical,
    commit_safetensors, sha256_file, verify_receipt)
from mistral_large4.prepare_corpus import load_token_records
from .source import BASE_REVISION, MistralSource, official_identity

SCHEMA = 'cuteafd-mistral-large4-core-pilot-v1'
PROOF_LAYERS = (0, 3, 17, 35, 39, 71)
WITNESS_LAYERS = (0, 3, 16, 17, 35, 39, 71)
WITNESS_TOKENS = 2205
GIB = 1024 ** 3


def candidate_specs(layer):
    """Twelve matrix geometries, sixteen quantizations; preserve indexer/kv_b."""
    prefix = f'layers.{layer}.'
    relative = {
        0: {'attention.wq': (6, 5, 4), 'attention.wkv_a_with_mqa': (6, 5),
            'attention.wo': (6,), 'feed_forward.w1': (6,), 'feed_forward.w2': (6,)},
        3: {'shared_experts.w1': (6,), 'shared_experts.w2': (6,), 'experts.$selected.w1': (4,)},
        17: {'attention.wo': (6,)},
        35: {'attention.wkv_a_with_mqa': (6,)},
        39: {'attention.wq': (6, 5)},
        71: {'attention.wo': (6,)},
    }.get(layer, {})
    return {prefix + name: bits for name, bits in relative.items()}


def select_records(calibration, heldout):
    if len(calibration) < 2 or len(heldout) < 2:
        raise ValueError('short pilot requires two complete prompts in each split')
    # Original order, no truncation, packing or calibration/heldout overlap.
    selected = {'calibration': calibration[:2], 'heldout': heldout[:2]}
    ids = [row['id'] for split in selected.values() for row in split]
    if len(set(ids)) != len(ids):
        raise ValueError('pilot prompt IDs overlap')
    if any(not row['input_ids'] for split in selected.values() for row in split):
        raise ValueError('pilot requires nonempty original prompts')
    return selected


def weighted_nmse(weight, candidate, hessian):
    """Heldout trace(E H E.T)/trace(W H W.T), FP64 final accumulation."""
    import torch
    if (weight.ndim != 2 or weight.shape != candidate.shape
            or tuple(hessian.shape) != (weight.shape[1], weight.shape[1])):
        raise ValueError('incompatible heldout projection metric geometry')
    numerator = denominator = 0.0
    # Bound the dense FFN matrix metric's temporary float32 storage.
    for start in range(0, weight.shape[0], 1024):
        reference = weight[start:start+1024].float()
        error = candidate[start:start+1024].float() - reference
        numerator += float(((error @ hessian) * error).sum(dtype=torch.float64))
        denominator += float(((reference @ hessian) * reference).sum(dtype=torch.float64))
    if (not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0
            or numerator < -1e-6 * denominator):
        raise ValueError('nonfinite or invalid heldout quadratic projection metric')
    return max(0.0, numerator) / denominator


def accepted(module, bits, nmse):
    if not math.isfinite(nmse) or nmse < 0:
        return False
    return nmse <= (0.05 if '.experts.' in module else 0.001) if bits == 6 or '.experts.' in module else True


def tp2_bytes(module, shape, bits):
    n, k = shape
    # Arithmetic projection for two rank shards; not a serving measurement.
    if module.endswith('wkv_a_with_mqa'):
        pass
    elif module.endswith(('.wo', '.w2')):
        k //= 2
    else:
        n //= 2
    stored = (n + 127) // 128 * 128
    size = (k * stored * bits // 8) + 2 * (k + stored) + 4
    return [size, size]


def compact():
    import torch
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class Progress:
    def __init__(self, root, budget):
        self.root, self.budget, self.started = Path(root), budget, time.monotonic()

    def check(self):
        if time.monotonic() - self.started > self.budget:
            raise TimeoutError('bounded pilot exhausted its 3600-second attempt budget; committed cache retained')

    def emit(self, event, **fields):
        import psutil
        import torch
        self.check()
        row = dict(time=datetime.now(timezone.utc).isoformat(), event=event,
            elapsed_seconds=time.monotonic()-self.started, rss_gib=psutil.Process().memory_info().rss/GIB,
            cuda_gib=torch.cuda.memory_allocated()/GIB, gpu_peak_gib=torch.cuda.max_memory_allocated()/GIB,
            free_gib=shutil.disk_usage(self.root).free/GIB, **fields)
        atomic_json(self.root/'status.json', row)
        with (self.root/'events.jsonl').open('a') as stream:
            stream.write(canonical(row)+'\n')
        print(canonical(row), flush=True)
        if row['rss_gib'] > 74 or row['cuda_gib'] > 86 or row['free_gib'] < 64:
            raise MemoryError('pilot exceeded host/GPU/disk admission')


def _load(path, device='cuda:0'):
    from safetensors.torch import load_file
    return {key: value.clone().to(device) for key, value in load_file(str(path)).items()}


def persistent(context, topk, state):
    result = dict(prev_topk_indices=topk)
    if 'position_ids' in state:
        result['position_ids'] = state['position_ids']
    shared = context['shared_kv_states']
    if set(shared) - {'nope'}:
        raise ValueError('unreviewed pilot KV sharing class')
    if 'nope' in shared:
        result['kv_nope_pass'], result['kv_nope_rot'] = shared['nope']
    return result


def reference_forward(block, hidden, rotary, state):
    from . import native
    context = native.prepare_context(hidden, rotary, state.get('prev_topk_indices'),
        shared_kv_states=native._shared_from_state(state), position_ids=state.get('position_ids'))
    output, topk = block(hidden, past_key_values=None, use_cache=False, **context)
    return output, persistent(context, topk, state)


def compare_states(actual, expected, label):
    import torch
    if set(actual) != set(expected) or any(not torch.equal(actual[k], expected[k]) for k in actual):
        raise ValueError('native/staged Mistral state differs: ' + label)


def staged_proof(block, hidden, rotary, state, expected, expected_state):
    import torch
    from . import native
    normalized = native.attention_inputs(block, hidden)
    o_input, actual_state = native.attention_capture(block, normalized, rotary, state)
    residual, normalized = native.attention_finish(block, hidden, block.self_attn.o_proj(o_input))
    actual = (residual + block.mlp(normalized) if block.layer_idx == 0
              else native.complete_moe(residual, normalized, block.mlp))
    if not torch.equal(actual, expected):
        raise ValueError(f'native/staged Mistral hidden bytes differ at layer {block.layer_idx}')
    compare_states(actual_state, expected_state, f'layer-{block.layer_idx}')


class Capture:
    """Actual source linear inputs, raw XTX; routed witness uses natural rows."""
    def __init__(self, block, layer, shared):
        self.block, self.layer, self.shared = block, layer, shared
        self.handles, self.split = [], None

    def __enter__(self):
        from . import native
        for name in candidate_specs(self.layer):
            if '$selected' in name:
                self.handles.append(self.block.mlp.gate.register_forward_hook(self.route_hook))
            else:
                module = native.linear_module(self.block, name)
                self.handles.append(module.register_forward_pre_hook(self.hook(name)))
        return self

    def add(self, name, value):
        from . import numerical
        key = (self.split, name)
        if value.numel() == 0:
            return
        if key not in self.shared['hessians']:
            self.shared['hessians'][key] = numerical.DenseRawHessian(value.shape[-1], device=value.device)
        self.shared['hessians'][key].add(value)

    def hook(self, name):
        def accumulate(module, arguments):
            self.add(name, arguments[0])
        return accumulate

    def route_hook(self, module, arguments, result):
        import torch
        _, weights, indices = result
        if 'expert' not in self.shared:
            if self.split != 'calibration':
                raise ValueError('routed pilot expert must be selected from calibration only')
            self.shared['expert'] = int(torch.bincount(indices.flatten(), minlength=module.num_experts).argmax())
        expert = self.shared['expert']
        value = arguments[0].reshape(-1, module.hidden_dim)
        natural = (indices == expert).any(dim=1)
        self.add(f'layers.{self.layer}.experts.{expert}.w1', value[natural])

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()


def saved_capture(root, layer, identity):
    path = root/f'capture/layer-{layer:03d}/manifest.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result['identity'] != identity or result['layer'] != layer:
        raise ValueError('pilot capture identity differs; cached observations require review')
    for entry in result['hessians']:
        verify_receipt(path.parent/entry['filename'], entry, checksum=True)
    return result


def commit_capture(root, layer, shared, proofs, identity):
    directory = root/f'capture/layer-{layer:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for (split, module), hessian in sorted(shared['hessians'].items()):
        raw = hessian.raw_dict()
        filename = f'{split}-{module}.safetensors'
        path = directory/filename
        # No manifest references this partial attempt; retain every committed
        # capture and replace only this layer's uncommitted orphan payloads.
        path.unlink(missing_ok=True)
        entries.append(dict(split=split, module=module, filename=filename, count=raw['count'],
            **commit_safetensors(path, dict(H=raw['H']))))
    expected = candidate_specs(layer)
    expert = shared.get('expert')
    expected = {name.replace('$selected', str(expert)) for name in expected}
    if {(entry['split'], entry['module']) for entry in entries} != {
            (split, name) for split in ('calibration', 'heldout') for name in expected}:
        raise ValueError('pilot has no natural heldout observations for a selected projection')
    manifest = dict(identity=identity, layer=layer, expert=expert, hessians=entries, proofs=proofs)
    atomic_json(directory/'manifest.json', manifest)
    return manifest


def run_candidates(source, root, capture, identity, progress):
    import torch
    from . import numerical
    layer = capture['layer']
    directory = root/f'capture/layer-{layer:03d}'
    candidates = root/'candidates'
    candidates.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = {name.replace('$selected', str(capture['expert'])): bits
             for name, bits in candidate_specs(layer).items()}
    for module, arms in specs.items():
        progress.check()
        descriptors = {entry['split']: entry for entry in capture['hessians'] if entry['module'] == module}
        calibration = descriptors['calibration']
        heldout = descriptors['heldout']
        raw = dict(H=_load(directory/calibration['filename'])['H'], count=calibration['count'],
                   finalized=False, identity_rows=0)
        weight = source.tensor(module+'.weight', 'cuda:0')
        shape = list(weight.shape)
        for bits in arms:
            progress.check()
            path = candidates/f'{module}.k{bits}.safetensors'
            receipt_path = path.with_suffix('.json')
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt['identity'] != identity or receipt['module'] != module or receipt['bits'] != bits
                        or receipt['calibration_sha256'] != calibration['sha256']):
                    raise ValueError('pilot serialized candidate identity differs; cache needs review')
                verify_receipt(path, receipt, checksum=True)
            else:
                path.unlink(missing_ok=True)
                packed, metrics, evidence = numerical.quantize_projection(weight, raw, bits=bits,
                    seed=787, sigma=.025, device='cuda:0')
                dense = numerical.reconstruct_projection(packed, shape, device='cuda:0', dtype=torch.bfloat16)
                receipt = dict(identity=identity, module=module, bits=bits, logical_shape=shape,
                    calibration_sha256=calibration['sha256'], metrics=metrics, evidence=evidence,
                    **commit_safetensors(path, packed))
                # Reconstruct once more from actual serialized tensors; memory
                # candidates never substitute for the checkpoint's FP16 scales.
                reloaded = numerical.reconstruct_projection(_load(path, 'cpu'), shape,
                    device='cuda:0', dtype=torch.bfloat16)
                if not torch.equal(dense, reloaded):
                    raise ValueError('serialized Mistral pilot reconstruction differs')
                atomic_json(receipt_path, receipt)
                del packed, dense, reloaded
            dense = numerical.reconstruct_projection(_load(path, 'cpu'), shape,
                device='cuda:0', dtype=torch.bfloat16)
            if dense.shape != weight.shape or not bool(torch.isfinite(dense).all()):
                raise ValueError('nonfinite/cropped Mistral serialized candidate')
            H = _load(directory/heldout['filename'])['H']
            nmse = weighted_nmse(weight, dense, H)
            packed_size = sum(value.numel()*value.element_size() for value in _load(path, 'cpu').values())
            row = dict(tensor=module, arm=f'mcg-k{bits}', bits=bits, nmse=nmse,
                passed=accepted(module, bits, nmse), calibration_rows=calibration['count'],
                heldout_rows=heldout['count'], payload_bytes=packed_size,
                bpw=packed_size*8/weight.numel(), tp2_bytes=tp2_bytes(module, shape, bits),
                logical_shape=shape, output_padding=receipt['evidence']['output_padding'],
                serialized_replay='passed')
            rows.append(row)
            progress.emit('pilot_result', layer=layer, **row)
            del dense, H
        source_dtype, _ = source.tensor_header(module+'.weight')
        source_bytes = weight.numel() * (1 if source_dtype == 'F8_E4M3' else 2)
        if source_dtype == 'F8_E4M3':
            _, grid = source.tensor_header(module+'.qscale_weight')
            source_bytes += math.prod(grid)*4
        rows.append(dict(tensor=module, arm='source-fp8' if source_dtype == 'F8_E4M3' else 'source-bf16',
            nmse=0.0, passed=True, control='identical-exact-source-reference',
            payload_bytes=source_bytes, bpw=source_bytes*8/weight.numel()))
        rows.append(dict(tensor=module, arm='dequantized-bf16', nmse=0.0, passed=True,
            control='same-reference-not-an-independent-higher-precision-checkpoint',
            payload_bytes=weight.numel()*2, bpw=16.0))
        del weight, raw
        source.close()
        compact()
    atomic_json(root/f'capture/layer-{layer:03d}/results.json', dict(identity=identity, results=rows))
    return rows


def initial(source, boundary, records, progress):
    import torch
    current = boundary.current(checksum=True)
    if current is not None:
        return current
    directory = boundary.directory(-1)
    directory.mkdir(parents=True, exist_ok=True)
    embedding = source.tensor('tok_embeddings.weight', 'cuda:0')
    entries = []
    for i, (split, row) in enumerate(records):
        path = directory/f'{i:02d}.safetensors'
        path.unlink(missing_ok=True)
        ids = torch.tensor(row['input_ids'], device='cuda:0')
        hidden = torch.nn.functional.embedding(ids, embedding).unsqueeze(0)
        entries.append(dict(filename=path.name, id=row['id'], split=split,
            **commit_safetensors(path, dict(hidden=hidden))))
    source.close()
    del embedding, hidden
    compact()
    progress.emit('pilot_embeddings_committed')
    return boundary.commit(-1, entries)


def source_replay(source, root, selected, identity, progress):
    import torch
    from . import native
    boundary = RollingBoundary(root/'frontier', identity)
    records = [(split, row) for split in ('calibration', 'heldout') for row in selected[split]]
    current = initial(source, boundary, records, progress)
    rotary = source.rotary('cuda:0')
    results, proofs = [], []
    # A numerical/OOM interruption can leave verified captures at the current
    # layer; they are reused, never deleted to force apparent progress.
    for layer in range(72):
        progress.check()
        capture = saved_capture(root, layer, identity) if layer in PROOF_LAYERS else None
        if layer <= current['layer']:
            if layer in PROOF_LAYERS:
                if capture is None:
                    raise ValueError('committed source frontier lost its pilot capture')
                path = root/f'capture/layer-{layer:03d}/results.json'
                saved = json.loads(path.read_text())
                if saved['identity'] != identity:
                    raise ValueError('committed pilot result identity differs')
                results.extend(saved['results'])
                proofs.extend(capture['proofs'])
            continue
        block = source.load_block(layer, 'cuda:0', include_experts=layer != 0)
        progress.emit('pilot_source_layer_loaded', layer=layer)
        shared = dict(hessians={})
        layer_proofs, entries = [], []
        successor = boundary.directory(layer)
        successor.mkdir(parents=True, exist_ok=True)
        for i, ((split, row), prior) in enumerate(zip(records, current['manifest']['batches'], strict=True)):
            progress.check()
            if prior['id'] != row['id'] or prior['split'] != split:
                raise ValueError('pilot source frontier corpus order differs')
            state = _load(boundary.root/current['directory']/prior['filename'])
            hidden = state.pop('hidden')
            if capture is None and layer in PROOF_LAYERS:
                with Capture(block, layer, shared) as hooks:
                    hooks.split = split
                    output, next_state = reference_forward(block, hidden, rotary, state)
                del hooks
            else:
                output, next_state = reference_forward(block, hidden, rotary, state)
            if not bool(torch.isfinite(output).all()):
                raise ValueError('nonfinite frozen-source Mistral pilot frontier')
            if layer in PROOF_LAYERS and capture is None:
                staged_proof(block, hidden, rotary, state, output, next_state)
                layer_proofs.append(dict(layer=layer, id=row['id'], split=split,
                    tokens=hidden.shape[1], hidden='byte-exact', reuse_state='byte-exact',
                    native_parameter_source='unchanged-official-source-FP8-dequantized-to-BF16'))
            path = successor/prior['filename']
            path.unlink(missing_ok=True)
            entries.append(dict(filename=path.name, id=row['id'], split=split,
                **commit_safetensors(path, dict(hidden=output, **next_state))))
            del hidden, output, state, next_state
        del block
        compact()
        if layer in PROOF_LAYERS:
            if capture is None:
                capture = commit_capture(root, layer, shared, layer_proofs, identity)
            del shared
            compact()
            results.extend(run_candidates(source, root, capture, identity, progress))
            proofs.extend(capture['proofs'])
            if any(not row['passed'] for row in results):
                raise ValueError('short Mistral pilot exceeded a declared heldout NMSE acceptance bound')
        current = boundary.commit(layer, entries)
        progress.emit('pilot_source_layer_committed', layer=layer, source_layers=72)
    return results, proofs, current['layer'] + 1


def sparse_witness(source, root, selected, identity, progress):
    import torch
    from . import native
    path = root/'sparse-witness.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['identity'] != identity or result['status'] != 'passed':
            raise ValueError('sparse witness identity differs')
        return result
    ids = [token for row in selected['calibration'] for token in row['input_ids']]
    ids = (ids * ((WITNESS_TOKENS+len(ids)-1)//len(ids)))[:WITNESS_TOKENS]
    embedding = source.tensor('tok_embeddings.weight', 'cuda:0')
    hidden = torch.nn.functional.embedding(torch.tensor(ids, device='cuda:0'), embedding).unsqueeze(0)
    del embedding
    rotary = source.rotary('cuda:0')
    state, rows = {}, []
    for layer in WITNESS_LAYERS:
        progress.check()
        block = source.load_block(layer, 'cuda:0', include_experts=False)
        normalized = native.attention_inputs(block, hidden)
        context = native.prepare_context(normalized, rotary, state.get('prev_topk_indices'),
            shared_kv_states=native._shared_from_state(state))
        expected, _, topk = block.self_attn(hidden_states=normalized, past_key_values=None, **context)
        expected_state = persistent(context, topk, state)
        value, actual_state = native.attention_capture(block, normalized, rotary, state)
        actual = block.self_attn.o_proj(value)
        if not torch.equal(actual, expected):
            raise ValueError(f'2205-token sparse native attention differs at layer {layer}')
        compare_states(actual_state, expected_state, f'sparse-witness-{layer}')
        if topk.shape[-1] != 2048:
            raise ValueError('synthetic witness did not enter the sparse top-2048 branch')
        rows.append(dict(layer=layer, tokens=WITNESS_TOKENS, topk_width=2048,
            attention_output='byte-exact', reuse_state='byte-exact',
            role='synthetic-embedding-input-native-correctness-only'))
        state = expected_state
        del block, normalized, context, expected, actual, value, topk
        compact()
        progress.emit('pilot_sparse_witness_passed', layer=layer)
    result = dict(identity=identity, status='passed', results=rows,
        claim='native sparse attention correctness; not full-model or long-context generation quality')
    atomic_json(path, result)
    return result


def make_identity(source, tokens, selected):
    code = [Path(__file__).with_name(name) for name in ('pilot.py', 'source.py', 'native.py', 'numerical.py', 'recipe.py')]
    code += [REPO/'quantization/mistral_large4/prepare_corpus.py',
             REPO/'quantization/kimi_k3/uniform_k2/storage.py']
    fork = REPO/'third_party/gptqmodel'
    code += [fork/'gptqmodel/utils'/name for name in ('glm5_core.py', 'kimi_k3_core_stream.py', 'kimi_k3_core_format.py')]
    return json.loads(canonical(dict(schema=SCHEMA, source_revision=BASE_REVISION,
        source_config_sha256=sha256_file(source.snapshot/'config.json'),
        source_index_sha256=sha256_file(source.snapshot/source.index_name),
        corpus_manifest_sha256=sha256_file(tokens/'manifest.json'),
        selected={split:dict(ids=[r['id'] for r in rows], tokens=sum(len(r['input_ids']) for r in rows),
            sha256=hashlib.sha256(canonical(rows).encode()).hexdigest()) for split, rows in selected.items()},
        source_trajectory='72-layer-frozen-source-causal-replay-no-candidate-installed',
        proof_layers=PROOF_LAYERS, witness_layers=WITNESS_LAYERS, witness_tokens=WITNESS_TOKENS,
        acceptance=dict(core_k6_nmse=0.001, routed_k4_nmse=0.05, other_arms='informational'),
        candidates={str(layer):candidate_specs(layer) for layer in PROOF_LAYERS},
        official=official_identity(), code={str(path.relative_to(REPO)):sha256_file(path) for path in code},
        gptqmodel_commit=subprocess.check_output(['git','-C',str(fork),'rev-parse','HEAD'], text=True).strip(),
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','safetensors')},
        seed=787, sigma_reg=.025, head='production-only; no full-model quality claim')))


def validate_summary_inputs(results, proofs, witness, selected, identity, source_layers):
    """Completion flags require the full declared proof and candidate inventory."""
    if source_layers != 72:
        raise ValueError('pilot has not completed all 72 frozen-source layers')
    expected_proofs = {(layer, split, row['id']): len(row['input_ids'])
                       for layer in PROOF_LAYERS for split, rows in selected.items() for row in rows}
    observed_proofs = {(row['layer'], row['split'], row['id']): row for row in proofs}
    if (len(proofs) != len(expected_proofs) or set(observed_proofs) != set(expected_proofs)
            or any(row.get('hidden') != 'byte-exact' or row.get('reuse_state') != 'byte-exact'
                   or row.get('tokens') != expected_proofs[key] for key, row in observed_proofs.items())):
        raise ValueError('pilot native byte-exact proof inventory is incomplete')
    witness_rows = witness.get('results', [])
    if (witness.get('identity') != identity or witness.get('status') != 'passed'
            or len(witness_rows) != len(WITNESS_LAYERS)
            or {row.get('layer') for row in witness_rows} != set(WITNESS_LAYERS)
            or any(row.get('tokens') != WITNESS_TOKENS or row.get('topk_width') != 2048
                   or row.get('attention_output') != 'byte-exact' or row.get('reuse_state') != 'byte-exact'
                   for row in witness_rows)):
        raise ValueError('pilot sparse native witness inventory is incomplete')
    routed = {row['tensor'] for row in results if '.experts.' in row['tensor']}
    if len(routed) != 1:
        raise ValueError('pilot routed candidate inventory differs')
    routed_name = next(iter(routed))
    fields = routed_name.split('.')
    if (len(fields) != 5 or fields[:3] != ['layers', '3', 'experts'] or fields[-1] != 'w1'
            or not fields[3].isdigit() or not 0 <= int(fields[3]) < 128):
        raise ValueError('pilot routed candidate is outside the reviewed source layer')
    specs = {name.replace('layers.3.experts.$selected.w1', routed_name): bits
             for layer in PROOF_LAYERS for name, bits in candidate_specs(layer).items()}
    expected = {(name, f'mcg-k{bits}') for name, arms in specs.items() for bits in arms}
    expected.update((name, 'dequantized-bf16') for name in specs)
    expected.update((name, 'source-bf16' if name.endswith('wkv_a_with_mqa') else 'source-fp8') for name in specs)
    observed = {(row['tensor'], row['arm']): row for row in results}
    if len(results) != len(expected) or set(observed) != expected:
        raise ValueError('pilot serialized candidate/control inventory is incomplete')
    for (name, arm), row in observed.items():
        if not row.get('passed') or not math.isfinite(row['nmse']) or row['nmse'] < 0:
            raise ValueError('pilot contains a failed or nonfinite candidate')
        if arm.startswith('mcg-k'):
            if (row.get('serialized_replay') != 'passed' or row.get('bits') != int(arm[-1])
                    or not accepted(name, row['bits'], row['nmse'])
                    or row.get('calibration_rows', 0) <= 0 or row.get('heldout_rows', 0) <= 0):
                raise ValueError('pilot candidate lacks successful serialized replay and heldout acceptance')
        elif row['nmse'] != 0:
            raise ValueError('identical source control has nonzero error')
    return dict(native_boundaries_byte_exact=True, serialized_projection_replay=True, source_layers=72)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot', 'tokens', 'root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.root, args.tokens, args.output = (path.expanduser().resolve() for path in (args.root,args.tokens,args.output))
    source = MistralSource(args.source_snapshot)
    if source.snapshot.name != BASE_REVISION:
        raise ValueError('pilot requires the pinned original Mistral FP8 source')
    source.native_config()
    selected = select_records(load_token_records(args.tokens, 'calibration'), load_token_records(args.tokens,'heldout'))
    identity = make_identity(source, args.tokens, selected)
    plan_path = args.root/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != identity:
        raise ValueError('pilot identity differs: review numerical changes before reusing any cached observations')
    if args.preflight:
        print(canonical(dict(admission='passed', budget_seconds=3600, identity=identity)), flush=True)
        return
    if plan_path.exists() and not args.resume:
        raise ValueError('existing short pilot requires explicit --resume')
    if args.resume and not plan_path.exists():
        raise ValueError('pilot resume requires a committed identity')
    args.root.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report['identity'] != identity or report['status'] != 'passed':
            raise ValueError('pilot output identity/status differs')
        qualification = validate_summary_inputs(report['results'],report['native_proofs'],report['sparse_witness'],
            selected,identity,report.get('source_layers'))
        if any(report.get(key) != value for key,value in qualification.items()):
            raise ValueError('pilot completion qualification flags differ')
        print(canonical(dict(status='passed', reused=True, output=str(args.output))), flush=True)
        return
    atomic_json(plan_path, identity)
    import torch
    from .run import exclusive_run
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86*GIB/torch.cuda.get_device_properties(0).total_memory)
    progress = Progress(args.root, 3600)
    try:
        with torch.inference_mode(), exclusive_run(args.root):
            progress.emit('pilot_started')
            witness = sparse_witness(source,args.root,selected,identity,progress)
            results, proofs, source_layers = source_replay(source,args.root,selected,identity,progress)
            qualification = validate_summary_inputs(results,proofs,witness,selected,identity,source_layers)
            report = dict(identity=identity, status='passed', results=results, native_proofs=proofs,
                sparse_witness=witness, elapsed_seconds=time.monotonic()-progress.started,
                **qualification,
                claim='local projection quality and exact native boundaries only; full-model quality unmeasured')
            atomic_json(args.output, report)
            progress.emit('pilot_complete', status='passed', output=str(args.output))
    except BaseException as error:
        atomic_json(args.root/'failure.json', dict(time=datetime.now(timezone.utc).isoformat(),
            type=type(error).__name__, error=str(error), cache_action='preserved-all-committed-work-for-diagnosis'))
        raise


if __name__ == '__main__':
    main()
