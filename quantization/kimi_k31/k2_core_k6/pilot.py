#!/usr/bin/env python3
"""Small native/staged and serialized format witnesses, never a prefix quant.

Layer 3 receives a labelled embedding-input witness, not an asserted causal
layer-3 stream. Full production captures its own full-panel causal inputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get('KIMI_GPTQMODEL_SOURCE', str(REPO / 'third_party/gptqmodel')))
sys.path.insert(0, str(REPO / 'quantization'))
if __package__ in (None, ''):
    __package__ = 'kimi_k31.k2_core_k6'

from kimi_k3.prepare_corpus import load_token_records, require_source
from kimi_k3.uniform_k2.storage import atomic_json, canonical, commit_safetensors, sha256_file

SCHEMA = 'cuteafd-kimi-k31-short-pilot-v1'
CHECKS = ('kda_dense_boundary_byte_exact', 'dsa_attention_boundary_byte_exact',
          'dsa_indexer_f32', 'hessian_fp64_oracle', 'serialized_k6_reload',
          'serialized_k2_gate_up_down_reload')


def code_identity(repo=REPO):
    """Only immutable numerical sources; launcher/publication changes are administrative."""
    paths = ['quantization/kimi_k3/prepare_corpus.py',
             'quantization/kimi_k31/k2_core_k6/pilot.py',
             'python/reference/families/kimi_k3/model.py',
             'python/reference/families/kimi_k3/k31.py',
             'python/reference/shape_invariant.py',
             'python/reference/families/mistral_large4/ops.py']
    for name in ('run.py', 'recipe.py'):
        path = 'quantization/kimi_k31/k2_core_k6/' + name
        if (repo / path).is_file():
            paths.append(path)
    return {path: sha256_file(repo / path) for path in paths}


def select_records(calibration, heldout, calibration_tokens=128, heldout_tokens=64):
    """Two actual, disjoint prompt prefixes; no padding or synthetic activations."""
    if (type(calibration_tokens) is not int or type(heldout_tokens) is not int
            or min(calibration_tokens, heldout_tokens) <= 0):
        raise ValueError('pilot token counts must be positive integers')
    left, right = calibration[0], heldout[0]
    if left['id'] == right['id']:
        raise ValueError('pilot calibration and heldout overlap')
    return (dict(id=left['id'], input_ids=left['input_ids'][:calibration_tokens]),
            dict(id=right['id'], input_ids=right['input_ids'][:heldout_tokens]))


def validate_qualification(report):
    if (report.get('schema') != SCHEMA or report.get('status') != 'passed'
            or any(report.get('native_checks', {}).get(key) is not True for key in CHECKS)
            or report.get('layers') != [0, 3] or len(report.get('results', [])) != 5
            or not report.get('identity', {}).get('code')):
        raise ValueError('short K3.1 pilot qualification is incomplete')
    import math
    for row in report['results']:
        if (row.get('bits') not in (2, 6) or row.get('serialized_reload_byte_exact') is not True
                or any(not isinstance(row.get(key), (float, int))
                       or not math.isfinite(row[key]) or row[key] < 0
                       for key in ('weight_nmse', 'heldout_output_nmse'))):
            raise ValueError('invalid short-pilot projection evidence')
    return report


def validate_report_code(report, repo=REPO):
    if report['identity']['code'] != code_identity(repo):
        raise ValueError('pilot numerical code differs from current producer')


def configuration(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-snapshot', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--calibration-tokens', type=int, default=128)
    parser.add_argument('--heldout-tokens', type=int, default=64)
    parser.add_argument('--baseline-snapshot', type=Path, default=Path('/mnt/sparknest/hf-home/hub') /
        'models--wrldsuksgo2mars--Kimi-K3-EXL3-K2-Core-K6/snapshots/1a28050d2954426a43a3db705e53ca120e5f2b47')
    parser.add_argument('--preflight', action='store_true')
    return parser.parse_args(argv)


def baseline_metrics(snapshot, module, bits):
    """Label K3 production training metrics separately from this heldout witness."""
    layer = int(module.split('.layers.')[1].split('.')[0])
    root = Path(snapshot) / 'quantization-errors'
    if bits == 6:
        path = root / 'core-k6' / f'layer-{layer:03d}' / (module + '.json')
        metrics = json.loads(path.read_text())['metrics']
    else:
        path = root / 'base-k2' / f'layer-{layer:03d}.json'
        rows = json.loads(path.read_text())['projections']
        matches = [row['metrics'] for row in rows if row['module'] == module]
        if len(matches) != 1:
            raise ValueError('K3 baseline projection inventory differs')
        metrics = matches[0]
    return dict(snapshot_revision=Path(snapshot).name, receipt_sha256=sha256_file(path),
        domain='K3-full-causal-panel-regularized-Hessian-training-not-this-pilot-heldout',
        hessian_sample_count=metrics['hessian_sample_count'],
        hessian_weighted_relative_error=metrics['hessian_weighted_relative_error'])


def main(argv=None):
    args = configuration(argv)
    require_source(args.tokens, args.source_snapshot)
    calibration, heldout = select_records(load_token_records(args.tokens),
        load_token_records(args.tokens, 'heldout'), args.calibration_tokens, args.heldout_tokens)
    identity = dict(source_revision=args.source_snapshot.name,
        source_config_sha256=sha256_file(args.source_snapshot / 'config.json'),
        source_index_sha256=sha256_file(args.source_snapshot / 'model.safetensors.index.json'),
        token_manifest_sha256=sha256_file(args.tokens / 'manifest.json'),
        code=code_identity(), gptqmodel_commit=subprocess.check_output(['git', '-C',
            str(REPO / 'third_party/gptqmodel'), 'rev-parse', 'HEAD'], text=True).strip(),
        packages={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'safetensors')},
        calibration=calibration, heldout=heldout,
        input_witness='embedding-input-format-and-native-staging-not-causal-layer3-calibration')
    if identity['packages']['transformers'] != '4.57.6':
        raise ValueError('K3.1 reference requires transformers==4.57.6')
    if args.preflight:
        print(canonical(dict(schema=SCHEMA, status='preflight-passed', identity=identity)), flush=True)
        return
    import numpy as np
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file
    from gptqmodel.utils import kimi_k31 as numerical
    from gptqmodel.utils.kimi_k3 import situ
    from kimi_k3.uniform_k2.run import quantize_projection
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    report = dict(schema=SCHEMA, status='running', identity=identity, layers=[0, 3],
                  native_checks={}, results=[], started=time.time())
    atomic_json(args.root / 'pilot-plan.json', report)
    source = numerical.KimiK31Source(args.source_snapshot, reference_root=REPO / 'python/reference')

    def clear():
        gc.collect()
        torch.cuda.empty_cache()

    def emit(event, **fields):
        print(canonical(dict(event=event, **fields)), flush=True)

    def quantify(module, weight, raw, held, bits):
        if bits == 6:
            packed, metrics, _ = numerical.quantize_core_projection(weight, raw)
        else:
            packed, metrics = quantize_projection(weight, raw)
        shape = list(weight.shape)
        native = numerical.reconstruct_core_projection(packed, shape)
        path = args.root / 'candidates' / (module + '.safetensors')
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt = commit_safetensors(path, packed)
        reloaded = load_file(str(path))
        if any(not torch.equal(packed[key].cpu(), reloaded[key]) for key in packed):
            raise ValueError('serialized packed bytes differ')
        dense = numerical.reconstruct_core_projection(reloaded, shape)
        if not torch.equal(native, dense):
            raise ValueError('serialized reconstructed matrix differs')
        expected, actual = F.linear(held, weight), F.linear(held, dense)
        row = dict(tensor=module, bits=bits, serialized_reload_byte_exact=True,
            weight_nmse=float((dense.float()-weight.float()).square().sum(dtype=torch.float64) /
                              weight.float().square().sum(dtype=torch.float64)),
            heldout_output_nmse=float((actual.float()-expected.float()).square().sum(dtype=torch.float64) /
                                      expected.float().square().sum(dtype=torch.float64)),
            error_metrics=metrics, receipt=receipt,
            k3_baseline=baseline_metrics(args.baseline_snapshot, module, bits))
        report['results'].append(row)
        atomic_json(args.root / 'results.json', report)
        emit('pilot_projection', **row)
        return dense

    with torch.inference_mode():
        embedding = source.tensor('language_model.model.embed_tokens.weight', 'cuda:0')
        hidden = F.embedding(torch.tensor(calibration['input_ids'], device='cuda:0'), embedding).unsqueeze(0)
        held = F.embedding(torch.tensor(heldout['input_ids'], device='cuda:0'), embedding).unsqueeze(0)
        del embedding
        clear()
        for layer in (0, 3):
            block = source.load_block(layer, include_experts=False)
            bank_count = 0 if layer == 0 else 1
            bank = hidden.reshape(-1, hidden.shape[-1]).unsqueeze(1).repeat(1, bank_count, 1)
            held_bank = held.reshape(-1, held.shape[-1]).unsqueeze(1).repeat(1, bank_count, 1)
            normalized, residual, next_bank = numerical.attention_inputs(block, hidden, bank)
            held_input, held_residual, held_next = numerical.attention_inputs(block, held, held_bank)
            name = 'q_proj' if layer == 0 else 'kv_a_proj_with_mqa'
            projection = getattr(block.self_attn, name)
            hessian = numerical.DenseRawHessian(normalized.shape[-1])
            hessian.add(normalized)
            raw = hessian.raw_dict()
            rows = normalized.reshape(-1, normalized.shape[-1]).double()
            oracle = rows.T @ rows
            relative = float((raw['H'].double()-oracle).norm()/oracle.norm())
            if relative >= 1e-6:
                raise ValueError('bounded covariance differs from FP64 oracle')
            report['native_checks']['hessian_fp64_oracle'] = True
            del oracle, rows, hessian
            dense = quantify(f'language_model.model.layers.{layer}.self_attn.{name}',
                             projection.weight, raw, held_input, 6)
            projection.weight = torch.nn.Parameter(dense, requires_grad=False)
            del dense, raw
            clear()
            actual_input = numerical.capture_linear_input(block.self_attn.o_proj, block.self_attn,
                hidden_states=normalized, attention_mask=None, use_cache=False)
            actual_attention = block.self_attn.o_proj(actual_input)
            expected_attention = block.self_attn(hidden_states=normalized, attention_mask=None, use_cache=False)
            if not torch.equal(actual_attention, expected_attention):
                raise ValueError('native/staged attention output differs')
            prefix, next_bank, ffn = numerical.attention_finish(block, residual, next_bank, actual_attention)
            if layer == 0:
                actual = prefix + block.mlp(ffn)
                expected, expected_bank = block(hidden, block_residual=bank, use_cache=False)
                if not torch.equal(actual, expected) or not torch.equal(next_bank, expected_bank):
                    raise ValueError('native/staged KDA dense boundary differs')
                report['native_checks']['kda_dense_boundary_byte_exact'] = True
                del actual, expected, expected_bank
            else:
                report['native_checks']['dsa_attention_boundary_byte_exact'] = True
                if any(parameter.dtype != torch.float32 for parameter in block.self_attn.indexer.parameters()):
                    raise ValueError('DSA indexer arithmetic is not F32')
                report['native_checks']['dsa_indexer_f32'] = True
                held_attention = block.self_attn(hidden_states=held_input, attention_mask=None, use_cache=False)
                _, _, held_ffn = numerical.attention_finish(block, held_residual, held_next, held_attention)
                moe = block.block_sparse_moe
                routes, _ = moe.gate(ffn)
                latent = moe.routed_expert_down_proj(ffn.reshape(-1, ffn.shape[-1]))
                held_latent = moe.routed_expert_down_proj(held_ffn.reshape(-1, held_ffn.shape[-1]))
                counts = torch.bincount(routes.reshape(-1), minlength=source.text_config['num_experts'])
                expert = int(counts.argmax())
                natural = torch.where((routes == expert).any(dim=-1))[0].cpu().numpy()
                spool = types.SimpleNamespace(inputs=latent, route_rows={expert: natural},
                    recovery_rows={expert: np.empty(0, dtype=np.int64)},
                    gather=lambda indices, device='cuda:0': latent[torch.as_tensor(indices, device=latent.device)].to(device))
                raw, _ = numerical.capture_routed_hessian(spool, expert)
                projections = {}
                for name in ('w1', 'w3', 'w2'):
                    module = f'language_model.model.layers.3.block_sparse_moe.experts.{expert}.{name}'
                    weight = source.decoded(module + '.weight', 'cuda:0')
                    held_rows = held_latent
                    if name == 'w2':
                        del raw
                        raw, _ = numerical.capture_routed_hessian(spool, expert, projections=projections)
                        held_rows = situ(F.linear(held_latent, projections['w1']),
                                         F.linear(held_latent, projections['w3'])).to(torch.bfloat16)
                    dense = quantify(module, weight, raw, held_rows, 2)
                    if name != 'w2':
                        projections[name] = dense
                    del dense, weight, held_rows
                    clear()
                del projections, raw, spool, latent, held_latent, held_ffn, held_attention
            report['native_checks']['serialized_k6_reload'] = True
            emit('pilot_native_passed', layer=layer)
            del block, normalized, held_input, residual, next_bank, held_residual, held_next
            del actual_input, actual_attention, expected_attention, prefix, ffn, projection, bank, held_bank
            clear()
        report['native_checks']['serialized_k2_gate_up_down_reload'] = True
        report.update(status='passed', seconds=time.time()-report['started'],
            cuda_peak_bytes=torch.cuda.max_memory_allocated())
        validate_qualification(report)
        atomic_json(args.root / 'qualification.json', report)
        emit('pilot_passed', seconds=report['seconds'])
    source.close()


if __name__ == '__main__':
    main()
