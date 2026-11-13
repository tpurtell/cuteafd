"""K3.1 calibrated dense-core-only K8 pass with native source expert replay."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import torch

from core_export import assemble_core_shard, finalize_core_export, sampled_packed_oracle
from kimi_k3.core_k6.run import ProjectionCheckpoints
from kimi_k3.uniform_k2.storage import atomic_json, sha256_file
from . import run
from .quality import LIMIT, training_error
from .recipe import reviewed_core_modules


def selection(bits=8):
    if type(bits) is not int or bits != 8:
        raise ValueError('isolated MC core production requires K8')
    return {name: dict(g, bits=bits) for name, g in reviewed_core_modules().items()}


class Checkpoints(ProjectionCheckpoints):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracles = {}

    def _evidence(self, module, saved):
        value = training_error(saved['metrics'])
        if value > LIMIT:
            raise ValueError(f'K8 calibrated training-error alarm: {module}: {value} > {LIMIT}')
        geometry = self.selection[module]
        if saved['evidence'].get('bits') != geometry['bits']:
            raise ValueError('core projection evidence bitrate differs')
        if module not in self.oracles:
            self.oracles[module] = sampled_packed_oracle(
                self.paths(module)[0], module, geometry, device=self.device)
        public = {key: saved[key] for key in ('module', 'logical_shape', 'metrics', 'evidence')}
        public['packed_byte_oracle'] = self.oracles[module]
        atomic_json(self.evidence_directory / (module + '.json'), public)


class Engine(run.Engine):
    def __init__(self, *args, numerical=None, **kwargs):
        if numerical is None:
            from gptqmodel.utils import kimi_k31, kimi_k3_core_stream, kimi_k3_core_format
            from types import SimpleNamespace
            names = ('attention_inputs', 'attention_finish', 'capture_linear_input',
                     'shared_down_input', 'routed_up_input', 'complete_moe', 'final_head_input')
            numerical = SimpleNamespace(**{name: getattr(kimi_k3_core_stream, name) for name in names},
                DenseRawHessian=kimi_k31.DenseRawHessian,
                quantize_core_projection=kimi_k3_core_format.quantize_core_projection,
                reconstruct_core_projection=kimi_k3_core_format.reconstruct_core_projection)
        super().__init__(*args, numerical=numerical, **kwargs)

    def _checkpoints(self, layer):
        name = 'head' if layer == 'head' else f'layer-{layer:03d}'
        directory = self.root / 'head/projections' if layer == 'head' else self.root / f'layers/{layer:03d}/projections'
        return Checkpoints(directory, self.root / 'evidence' / name, self.identity,
                           self.selection, self.progress, device=self.device, numerical=self.math)

    def _experts(self, block, store):
        # CoreEngine._phase_e loads the original MXFP4 experts before replay.
        # No expert Hessian, quantization, payload or route spool is produced.
        return None

    def _assemble_layer(self, layer, replacements):
        modules = {m: self.selection[m] for m in replacements}
        receipt = assemble_core_shard(replacements, modules, self.export,
            f'core-k8-layer-{layer:03d}.safetensors', self.identity)
        # Inherited recovery verifies this before retiring transient payloads.
        atomic_json(self.export / f'layer-{layer:03d}-receipt.json', receipt)
        return receipt

    def _finalize_export(self, replacements):
        selected = {m: self.selection[m] for m in replacements}
        assemble_core_shard(replacements, selected, self.export, 'core-k8-head.safetensors', self.identity)
        validate_evidence(self.selection, self.root)
        return finalize_core_export(self.selection, self.export, self.identity)


def validate_evidence(modules, root):
    for module, geometry in modules.items():
        directory = 'head' if module == 'language_model.lm_head' else 'layer-' + module.split('.')[3].zfill(3)
        row = json.loads((Path(root) / 'evidence' / directory / (module + '.json')).read_text())
        if (row.get('module') != module or row.get('logical_shape') !=
                [geometry['out_features'], geometry['in_features']]
                or training_error(row.get('metrics', {})) > LIMIT
                or row.get('evidence', {}).get('bits') != geometry['bits']):
            raise ValueError('core finalization reconstruction evidence differs')
        oracle = row.get('packed_byte_oracle', {})
        if (oracle.get('passed') is not True or oracle.get('bits') != geometry['bits']
                or oracle.get('schema') != 'cuteafd-core-packed-byte-oracle-v1'):
            raise ValueError('core finalization requires packed-byte oracle evidence')


def make_plan(source, tokens, root, export, bits):
    modules = selection(bits)
    root, export = Path(root).resolve(), Path(export).resolve()
    if root == export or root in export.parents or export in root.parents:
        raise ValueError('core run and export paths must be disjoint')
    records = run.load_token_records(tokens)
    if len(records) != 1426:
        raise ValueError('full K3.1 calibration corpus required')
    plan = copy.deepcopy(run.make_plan(source, tokens, root, export))
    recipe = copy.deepcopy(plan['identity']['recipe'])
    recipe.update(core_bits=bits, expert_bits=None, experts='original-native-source-unchanged',
                  causal_order='K8-core-serialized-replay-native-source-experts', core_only=True)
    recipe.pop('k6_training_error_alarm', None)
    recipe['training_error_alarm'] = dict(limit=LIMIT, domain='regularized-Hessian-training-not-heldout')
    plan['identity'] = copy.deepcopy(plan['identity'])
    plan['identity'].update(schema='cuteafd-kimi-k31-core-only-v1', recipe=recipe)
    plan['identity']['precision'] = dict(matmul_precision="highest", allow_tf32=False,
            torch_allow_tf32_cublas_override=os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            nvidia_tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"))
    plan['identity']['memory'] = dict(container_bytes=110 * run.GIB, cuda_allocation_cap_bytes=100 * run.GIB)
    for path in (Path(__file__), run.REPO / 'quantization/core_export.py'):
        plan['identity']['code'][str(path.relative_to(run.REPO))] = sha256_file(path)
    # Preserve full source validation, but reserve only isolated dense payloads.
    output = sum(g['in_features'] * g['stored_out_features'] * bits // 8
                 + (g['in_features'] + g['stored_out_features']) * 2 + 4 for g in modules.values())
    plan.update(core_only=True, core_bits=bits, publication_repo=None,
                inventory=dict(core_modules=len(modules), core_payload_bytes=output),
                required_free_bytes=output + plan['temporary_reserve_bytes'] + 64 * run.GIB)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot', 'tokens', 'root', 'export'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--core-only', action='store_true', required=True)
    parser.add_argument('--core-bits', type=int, default=8, choices=(8,))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    from gptqmodel.utils.kimi_k31 import KimiK31Source
    source = KimiK31Source(args.source_snapshot, reference_root=run.REPO / 'python/reference')
    plan = make_plan(source, args.tokens, args.root, args.export, args.core_bits)
    base = args.root
    while not base.exists():
        base = base.parent
    if not (args.root / 'plan.json').exists() and shutil.disk_usage(base).free < plan['required_free_bytes']:
        raise OSError('insufficient NVMe space for isolated core and rolling calibration banks')
    if args.preflight:
        source.close()
        print(run.canonical(plan), flush=True)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    args.export.mkdir(parents=True, exist_ok=True)
    with run.exclusive_run(args.root):
        saved = args.root / 'plan.json'
        if saved.exists():
            if not args.resume or json.loads(saved.read_text()) != plan:
                raise ValueError('existing core-only run requires resume and identical plan')
        else:
            if args.resume:
                raise FileNotFoundError('core-only resume requires producer plan')
            if any(args.export.iterdir()):
                raise ValueError('new core-only export must start empty')
            atomic_json(saved, plan)
        complete = args.root / 'quantization-complete.json'
        if complete.exists():
            if json.loads(complete.read_text())['identity'] != plan['identity']:
                raise ValueError('core-only completion identity differs')
            validate_evidence(selection(args.core_bits), args.root)
            finalize_core_export(selection(args.core_bits), args.export, plan['identity'])
            source.close()
            return
        progress = run.CoreProgress(args.root, host_limit=100, gpu_limit=100)
        progress.state['total_projections'] = len(selection(args.core_bits))
        try:
            torch.set_num_threads(8)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            torch.cuda.set_per_process_memory_fraction(100 * run.GIB / torch.cuda.get_device_properties(0).total_memory)
            records = run.load_token_records(args.tokens)
            engine = Engine(source, records, args.root, args.export, plan['identity'],
                            selection(args.core_bits), progress)
            current = engine.initial()
            engine.retire_committed(current)
            for layer in range(current['layer'] + 1, 93):
                current = engine.layer(layer, current)
            engine.head(current)
        except BaseException as error:
            progress.state.update(status='failed', error=repr(error), time=run.utc_now())
            atomic_json(args.root / 'status.json', progress.state)
            raise
        finally:
            source.close()
