#!/usr/bin/env python3
"""Layer-streamed Kimi K3.1 K2 experts / K6 core with rolling recovery."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'third_party/gptqmodel'))
sys.path.insert(0, str(REPO / 'quantization'))
if __package__ in (None, ''):
    __package__ = 'kimi_k31.k2_core_k6'
os.environ.setdefault('FLA_DISABLE_TENSOR_CACHE', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from mistral_large4.core_k6.numerical import fork_module
# Keep the HF4 reference separate from unrelated HF5 whole-model wrappers.
fork_module('utils.kimi_k31')
from kimi_k3.prepare_corpus import load_token_records
from kimi_k3.uniform_k2 import run as experts
from kimi_k3.uniform_k2.storage import (
    atomic_json, canonical, commit_safetensors, file_identity,
    fsync_directory, sha256_file, verify_receipt,
)
from kimi_k3.core_k6.run import (
    CoreEngine, Progress as CoreProgress, ProjectionCheckpoints,
    boundary_batch, compact_memory, exclusive_run, hardlink_frontier, utc_now,
)
from kimi_k3.core_k6.storage import StageStore
from .recipe import BASE_REPO, BASE_REVISION, TARGET_REPO, inventory, selected_modules
from .quality import QualityCheckpoints, LIMIT, BASELINE_REVISION

GIB = 1024 ** 3
RECIPE = dict(core_bits=6, expert_bits=2, seed=787, sigma_reg=.025,
    codebook='mcg', multiplier='0xCBAC1FED', out_scales='auto',
    hessian_accumulation='bounded-k128-fp32-products-fp64-outer-sum-fp32-storage',
    hessian='signed-block-hadamard-congruence-fp64-v1', core_identity_rows=0,
    cold_experts=dict(minimum=1024, ranks=[17, 32], order='rank-major-then-corpus-row',
                      residual='missing-rows-times-identity'),
    causal_order='attention-K6-shared-entry-and-latent-down-K6-experts-K2-shared-down-and-latent-up-K6',
    replay='serialized-exl3-fp16-scales-reconstructed-BF16',
    frontier='prefix-sum-and-complete-attention-residual-bank',
    dsa='shared-reference-eval-batch1-no-cache-f32-indexer')
RECIPE['k6_training_error_alarm'] = dict(limit=LIMIT, baseline_revision=BASELINE_REVISION,
                                       domain='regularized-Hessian-training-not-heldout')


class Progress(CoreProgress):
    def __init__(self, root):
        super().__init__(root, host_limit=86, gpu_limit=86)


def make_plan(source, tokens, root, export):
    summary = inventory(source)
    manifest = json.loads((tokens / 'manifest.json').read_text())
    if (source.snapshot.name != BASE_REVISION
            or manifest['source']['revision'] != BASE_REVISION
            or manifest['splits']['calibration']['count'] != 1426):
        raise ValueError('pinned source and full K3.1-rendered calibration required')
    if importlib.metadata.version('transformers') != '4.57.6':
        raise ValueError('K3.1 reference requires transformers==4.57.6')
    files = [*Path(__file__).parent.glob('*.py'),
        *REPO.joinpath('quantization/kimi_k3/core_k6').glob('*.py'),
        *REPO.joinpath('quantization/kimi_k3/uniform_k2').glob('*.py'),
        REPO / 'quantization/kimi_k3/prepare_corpus.py',
        *REPO.joinpath('python/reference/families/kimi_k3').glob('*.py'),
        REPO / 'python/reference/families/mistral_large4/ops.py',
        REPO / 'python/reference/shape_invariant.py']
    files += [REPO / 'quantization/mistral_large4/core_k6/numerical.py']
    # Operational launchers may be repaired without changing numerical identity.
    files = [p for p in files if p.name not in ('launch.py', 'container.py', 'publish.py',
                                               'pilot.py', 'preflight.py', 'probe.py')]
    fork = REPO / 'third_party/gptqmodel'
    files += [fork / ('gptqmodel/utils/' + name) for name in
              ('kimi_k31.py', 'kimi_k3.py', 'kimi_k3_core_stream.py',
               'kimi_k3_core_format.py', 'exl3_hessian.py')]
    identity = dict(schema='cuteafd-kimi-k31-k2-core-k6-v1',
        source_repo=BASE_REPO, source_revision=BASE_REVISION,
        source_config_sha256=sha256_file(source.snapshot / 'config.json'),
        source_index_sha256=sha256_file(source.snapshot / 'model.safetensors.index.json'),
        reference=source.reference_identity,
        token_manifest_sha256=sha256_file(tokens / 'manifest.json'), recipe=RECIPE,
        code={str(p.relative_to(REPO)): sha256_file(p) for p in sorted(set(files))},
        gptqmodel_commit=subprocess.check_output(['git', '-C', str(fork), 'rev-parse', 'HEAD'], text=True).strip(),
        packages={name: importlib.metadata.version(name) for name in
                  ('torch', 'transformers', 'fla-core', 'numpy', 'safetensors', 'huggingface-hub', 'tokenizers')},
        calibration=dict(prompts=1426, tokens=manifest['splits']['calibration']['tokens']))
    n = manifest['splits']['calibration']['tokens']
    frontier = n * 7168 * 2 * 9
    temporary = 3 * frontier + n * (7168 * 2 + 3584 * 2 + 16 * 10) + 48 * GIB
    output = summary['target_payload_bytes']
    return dict(identity=identity, source=str(source.snapshot), token_root=str(tokens),
        root=str(root), export=str(export), inventory=summary,
        source_fingerprints={s: file_identity(source.snapshot / s) for s in sorted(set(source.weight_map.values()))},
        frontier_max_bytes=frontier, temporary_reserve_bytes=temporary,
        required_free_bytes=output + temporary + 64 * GIB, publication_repo=TARGET_REPO)


class Engine(CoreEngine):
    def __init__(self, source, records, root, export, identity, selection, progress, *, device='cuda:0', numerical=None):
        if numerical is None:
            from gptqmodel.utils import kimi_k31 as numerical
        core = {name: spec for name, spec in selection.items() if '.experts.' not in name}
        super().__init__(source, records, root, export, identity, core, progress,
                         device=device, numerical=numerical)
        self.all_selection = selection

    def _checkpoints(self, layer):
        name = 'head' if layer == 'head' else f'layer-{layer:03d}'
        directory = self.root / 'head/projections' if layer == 'head' else self.root / f'layers/{layer:03d}/projections'
        return QualityCheckpoints(directory, self.root / 'evidence' / name, self.identity,
                                  self.selection, self.progress, device=self.device, numerical=self.math)

    def _capture_routes(self, block, store):
        directory = self.root / f'layers/{block.layer_idx:03d}/routes'
        receipt = directory / 'capture.json'
        if receipt.exists():
            manifest = json.loads(receipt.read_text())
            if manifest['identity'] != self.identity or manifest['layer'] != block.layer_idx:
                raise ValueError('route spool identity differs')
            for name, entry in manifest['files'].items():
                verify_receipt(directory / name, entry, checksum=True)
            return experts.RouteSpool(directory, manifest)
        directory.mkdir(parents=True, exist_ok=True)
        shapes = dict(latent=(self.tokens, 3584), routes=(self.tokens, 16),
                      weights=(self.tokens, 16), candidates=(16, self.tokens))
        types = dict(latent=np.uint16, routes=np.int16, weights=np.float32, candidates=np.int16)
        names = dict(latent='latent.bf16', routes='routes.i16', weights='weights.f32', candidates='candidates.i16')
        maps = {key: np.memmap(directory / names[key], mode='w+', dtype=types[key], shape=shape)
                for key, shape in shapes.items()}
        offset = 0
        try:
            moe = block.block_sparse_moe
            for index, row, state in self._read_batches(store, 'd'):
                flat = state['hidden'].reshape(-1, 7168)
                selected, weights = moe.gate(state['hidden'])
                latent = moe.routed_expert_down_proj(flat)
                scores = F.linear(flat.float(), moe.gate.weight.float()).sigmoid()
                scores.add_(moe.gate.e_score_correction_bias.unsqueeze(0))
                scores.scatter_(1, selected, float('-inf'))
                near = scores.topk(16, dim=-1, sorted=True).indices
                end = offset + len(flat)
                for key, value in dict(latent=latent.view(torch.uint16), routes=selected.short(),
                                       weights=weights.float()).items():
                    maps[key][offset:end] = value.cpu().numpy()
                maps['candidates'][:, offset:end] = near.cpu().short().numpy().T
                offset = end
                self._emit_batch('routes', block.layer_idx, index)
                del state, flat, selected, weights, latent, scores, near, value
            if offset != self.tokens:
                raise ValueError('route spool lost calibration rows')
            files = {}
            for key, mapping in maps.items():
                mapping.flush()
                mapping._mmap.close()
                path = directory / names[key]
                with path.open('rb') as file:
                    os.fsync(file.fileno())
                files[names[key]] = dict(sha256=sha256_file(path), file_identity=file_identity(path))
            maps.clear()
            manifest = dict(identity=self.identity, layer=block.layer_idx, tokens=self.tokens, files=files)
            atomic_json(receipt, manifest)
        finally:
            for mapping in maps.values():
                if not mapping._mmap.closed:
                    mapping._mmap.close()
        return experts.RouteSpool(directory, manifest)

    @torch.inference_mode()
    def _experts(self, block, store):
        layer = block.layer_idx
        directory = self.root / f'layers/{layer:03d}/expert-projections'
        directory.mkdir(parents=True, exist_ok=True)
        self.progress.emit('source_experts_load_started', layer=layer)
        self.source.materialize_routed_experts(block, self.device)
        self.progress.emit('source_experts_loaded', layer=layer)
        spool = self._capture_routes(block, store)
        started = time.monotonic()
        try:
            for expert, module in enumerate(block.block_sparse_moe.experts):
                raw = evidence = None
                for part in ('w1', 'w3', 'w2'):
                    name = f'language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{part}'
                    path = directory / (name + '.safetensors')
                    receipt_path = path.with_suffix('.json')
                    if receipt_path.exists():
                        saved = json.loads(receipt_path.read_text())
                        if saved['identity'] != self.identity or saved['module'] != name:
                            raise ValueError('expert checkpoint identity differs')
                        verify_receipt(path, saved, checksum=True)
                        packed = {k.removeprefix(name + '.'): v for k, v in load_file(str(path)).items()}
                    else:
                        path.unlink(missing_ok=True)
                        if part == 'w2':
                            hessian, proof = self.math.capture_routed_hessian(spool, expert,
                                projections={p: getattr(module, p).weight for p in ('w1', 'w3')}, device=self.device)
                        else:
                            if raw is None:
                                raw, evidence = self.math.capture_routed_hessian(spool, expert, device=self.device)
                            hessian, proof = raw, evidence
                        packed, metrics = experts.quantize_projection(getattr(module, part).weight, hessian)
                        saved = dict(identity=self.identity, module=name, logical_shape=list(getattr(module, part).weight.shape),
                            evidence=proof, metrics=metrics,
                            **commit_safetensors(path, {name + '.' + key: value for key, value in packed.items()}))
                        atomic_json(receipt_path, saved)
                        self.progress.emit('projection_committed', layer=layer, expert=expert, projection=part,
                            hessian_error=metrics['hessian_weighted_relative_error'], natural_rows=proof['natural_rows'])
                        del hessian
                    evidence_dir = self.root / 'evidence' / f'layer-{layer:03d}'
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    atomic_json(evidence_dir / (name + '.json'),
                                {key: saved[key] for key in ('module', 'logical_shape', 'metrics', 'evidence')})
                    dense = experts.reconstruct_exl3_tensors(packed, device=self.device, dtype=torch.bfloat16).T.contiguous()
                    getattr(module, part)._parameters['weight'] = nn.Parameter(dense, requires_grad=False)
                    del dense, packed, saved
                del raw, evidence
                if expert % 16 == 0 or expert == 895:
                    compact_memory()
                    seconds = (time.monotonic() - started) / (expert + 1)
                    self.progress.emit('expert_progress', layer=layer, expert=expert + 1, experts=896,
                                       seconds_per_expert=seconds,
                                       quantization_eta_seconds=seconds * ((92 - layer) * 896 + 895 - expert))
        finally:
            spool.close()

    def _execute_layer(self, block, current, store, checkpoints, phases):
        layer = block.layer_idx
        prefix = f'language_model.model.layers.{layer}.'
        attn = prefix + 'self_attn.'
        if phases['d'] is None and phases['e'] is None:
            if phases['c'] is None:
                if phases['a'] is None:
                    self._phase_a(block, current, store)
                leaves = ('q_proj', 'k_proj', 'v_proj', 'g_proj') if block.is_linear_attn else ('q_a_proj', 'kv_a_proj_with_mqa', 'g_proj')
                self._quant_group(block, checkpoints, [attn + n for n in leaves], store, 'a', 'input')
                if not block.is_linear_attn:
                    if phases['b'] is None and checkpoints.load(attn + 'q_b_proj') is None:
                        self._phase_b(block, store)
                    self._quant_group(block, checkpoints, [attn + 'q_b_proj'], store, 'b', 'q-b')
                self._phase_c(block, store)
            self._quant_group(block, checkpoints, [attn + 'o_proj'], store, 'c', 'o')
            self._phase_d(block, current, store)
        if layer:
            moe = prefix + 'block_sparse_moe.'
            if phases['e'] is None:
                self._quant_group(block, checkpoints, [moe + n for n in
                    ('shared_experts.gate_proj', 'shared_experts.up_proj', 'routed_expert_down_proj')], store, 'd', 'input')
                # The up-projection Hessian must see the completed K2 experts.
                self._experts(block, store)
                self._phase_e(block, store)
            self._quant_group(block, checkpoints, [moe + 'shared_experts.down_proj'], store, 'e', 'shared-down')
            self._quant_group(block, checkpoints, [moe + 'routed_expert_up_proj'], store, 'e', 'routed-up')
        self._phase_f(block, store)

    @torch.inference_mode()
    def layer(self, layer, current):
        from .export import assemble_layer
        started = time.monotonic()
        store = StageStore(self.root / f'layers/{layer:03d}/stages', self.identity)
        phases = {name: store.load(name, checksum=True) for name in ('a', 'b', 'c', 'd', 'e', 'f')}
        checkpoints = self._checkpoints(layer)
        prefix = f'language_model.model.layers.{layer}.'
        modules = [name for name in self.selection if name.startswith(prefix)]
        if phases['f'] is None:
            self.progress.emit('layer_load_started', layer=layer)
            block = self.source.load_block(layer, self.device, include_experts=False)
            for name in modules:
                checkpoints.install(name, block.get_submodule(self._local_name(layer, name)))
            self.progress.emit('layer_loaded', layer=layer)
            self._execute_layer(block, current, store, checkpoints, phases)
            del block
            self.source.close()
            compact_memory()
        replacements = checkpoints.replacements(modules)
        expert_dir = self.root / f'layers/{layer:03d}/expert-projections'
        for name in self.all_selection:
            if name.startswith(prefix) and '.experts.' in name:
                path = expert_dir / (name + '.safetensors')
                saved = json.loads(path.with_suffix('.json').read_text())
                if saved['identity'] != self.identity or saved['module'] != name:
                    raise ValueError('expert export checkpoint identity differs')
                verify_receipt(path, saved, checksum=True)
                replacements[name] = path
        self.progress.emit('layer_assembly_started', layer=layer)
        receipt = assemble_layer(self.source, layer, replacements, self.export, self.identity)
        batches = hardlink_frontier(self.boundary, layer, store.directory('f'), store.load('f'))
        current = self.boundary.commit(layer, batches)
        store.clear()
        checkpoints.retire_payloads()
        self._retire_expert_payloads(layer)
        elapsed = time.monotonic() - started
        self.layer_seconds.append(elapsed)
        atomic_json(self.root / f'layers/{layer:03d}/complete.json',
                    dict(identity=self.identity, layer=layer, seconds=elapsed, receipt=receipt, time=utc_now()))
        recent = self.layer_seconds[-6:]
        # Dense layer zero is not a useful estimate for MoE layers.
        remaining = sum(recent) / len(recent) * (92 - layer) if layer else None
        self.progress.emit('layer_committed', layer=layer, seconds=elapsed,
                           remaining_layers=92 - layer, layer_eta_seconds=remaining)
        return current

    def _retire_expert_payloads(self, layer):
        root = self.root / f'layers/{layer:03d}'
        for path in (root / 'expert-projections').glob('*.safetensors'):
            path.unlink()
        routes = root / 'routes'
        if routes.exists():
            if routes.is_symlink():
                raise ValueError('symlink route retirement')
            shutil.rmtree(routes)

    def retire_committed(self, current):
        super().retire_committed(current)
        for layer in range(current['layer'] + 1):
            self._retire_expert_payloads(layer)

    @torch.inference_mode()
    def head(self, current):
        from .export import finalize_export
        store = StageStore(self.root / 'head/stages', self.identity)
        checkpoints = self._checkpoints('head')
        module = 'language_model.lm_head'
        if checkpoints.load(module) is None:
            if store.load('head', checksum=True) is None:
                self._new_phase(store, 'head')
                reference = self.source.reference
                proj = nn.Linear(7168, 1, bias=False, device='meta')
                proj.weight = nn.Parameter(self.source.tensor('language_model.model.output_attn_res_proj.weight', self.device), requires_grad=False)
                norms = [reference.KimiRMSNorm(7168) for _ in range(2)]
                for norm, leaf in zip(norms, ('output_attn_res_norm', 'norm'), strict=True):
                    norm.weight = nn.Parameter(self.source.tensor(f'language_model.model.{leaf}.weight', self.device), requires_grad=False)
                    norm.variance_epsilon = self.source.text_config['rms_norm_eps']
                h = self._hessian(7168)
                for index, _ in enumerate(self.records):
                    state = boundary_batch(self.boundary, current, index, self.device)
                    hidden = self.math.final_head_input(state['prefix_sum'], state['block_residual'],
                        output_attn_res_proj=proj, output_attn_res_norm=norms[0], model_norm=norms[1],
                        apply_attn_res=reference._apply_attn_res)
                    h.add(hidden)
                    self._emit_batch('head', 93, index)
                    del state, hidden
                self._commit_phase(store, 'head', [], [('input', h)])
                del h, proj, norms, norm
                compact_memory()
            raw = self._raw(store, 'head', 'input')
            weight = self.source.tensor(module + '.weight', 'cpu')
            checkpoints.ensure(module, weight, raw)
            del weight, raw
            self.source.close()
            compact_memory()
        summary = finalize_export(self.source, self.all_selection, checkpoints.replacements([module]),
                                  self.export, self.identity, self.root / 'evidence')
        atomic_json(self.root / 'quantization-complete.json',
                    dict(identity=self.identity, summary=summary, time=utc_now()))
        store.clear()
        checkpoints.retire_payloads()
        shutil.rmtree(self.boundary.root)
        self.progress.state['status'] = 'quantized'
        self.progress.emit('quantization_complete', layer=93, **summary)
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot', 'tokens', 'root', 'export'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--publish-only', action='store_true')
    args = p.parse_args()
    from gptqmodel.utils.kimi_k31 import KimiK31Source
    source = KimiK31Source(args.source_snapshot, reference_root=REPO / 'python/reference')
    plan = make_plan(source, args.tokens, args.root, args.export)
    if args.preflight:
        base = args.root
        while not base.exists():
            base = base.parent
        if not (args.root / 'plan.json').exists() and shutil.disk_usage(base).free < plan['required_free_bytes']:
            raise OSError('insufficient NVMe space for output plus bounded rolling banks')
        print(canonical(plan), flush=True)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    args.export.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.root):
        saved = args.root / 'plan.json'
        if saved.exists():
            if not args.resume and not args.publish_only:
                raise FileExistsError('existing run requires --resume')
            if json.loads(saved.read_text()) != plan:
                raise ValueError('numerical/source/token identity changed; refuse recovery')
        else:
            if args.resume or args.publish_only:
                raise FileNotFoundError('recovery requires an existing producer plan')
            if shutil.disk_usage(args.root).free < plan['required_free_bytes']:
                raise OSError('insufficient NVMe space for rolling production')
            atomic_json(saved, plan)
        complete = args.root / 'quantization-complete.json'
        if args.publish_only:
            from .publish import publish
            record = json.loads(complete.read_text())
            if record['identity'] != plan['identity']:
                raise ValueError('publication identity differs')
            publish(args.export, args.root, TARGET_REPO)
            return
        if complete.exists():
            if json.loads(complete.read_text())['identity'] != plan['identity']:
                raise ValueError('complete producer identity differs')
            return
        progress = Progress(args.root)
        try:
            torch.set_num_threads(8)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(86 * GIB / torch.cuda.get_device_properties(0).total_memory)
            records = load_token_records(args.tokens)
            selection = selected_modules(source)
            engine = Engine(source, records, args.root, args.export, plan['identity'], selection, progress)
            if args.resume:
                engine.boundary.current(checksum=True)
            current = engine.initial()
            engine.retire_committed(current)
            for layer in range(current['layer'] + 1, 93):
                current = engine.layer(layer, current)
            engine.head(current)
        except BaseException as error:
            progress.state.update(status='failed', error=repr(error), time=utc_now())
            atomic_json(args.root / 'status.json', progress.state)
            raise
        finally:
            source.close()


if __name__ == '__main__':
    main()
