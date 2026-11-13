#!/usr/bin/env python3
"""One frozen, receipt-backed K8-causal pass into a directly written MC package."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from kimi_k3.uniform_k2.storage import atomic_json, canonical, commit_safetensors, verify_receipt, file_identity
from kimi_k3.core_k6.storage import StageStore
from glm5.core_k6.storage import ProjectionStore
from . import numerical, recipe
from .export import DirectMCWriter, VIEWS
from .prepare_corpus import load_token_batches
from .source import HeaderSource

REPO = Path(__file__).resolve().parents[2]
NUMERICAL_FILES = ('quantization/inkling2/run.py', 'quantization/inkling2/numerical.py',
    'quantization/inkling2/storage.py', 'quantization/inkling2/recipe.py',
    'quantization/inkling2/source.py', 'quantization/inkling2/prepare_corpus.py',
    'quantization/glm5/core_k6/storage.py', 'quantization/kimi_k3/core_k6/storage.py',
    'quantization/kimi_k3/uniform_k2/storage.py', 'quantization/kimi_k3/prepare_corpus.py')


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def git(*args, cwd=REPO):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def plan(source, corpus):
    if os.environ.get('INKLING_GPTQMODEL_ROOT'):
        raise ValueError('production requires the committed engine submodule pin')
    fork = REPO/'third_party/gptqmodel'
    pin = git('rev-parse', 'HEAD', cwd=fork)
    if git('status', '--porcelain', cwd=fork):
        raise ValueError('production quantizer checkout is dirty')
    if git('ls-tree', 'HEAD', 'third_party/gptqmodel').split()[2] != pin:
        raise ValueError('production fork differs from engine pin')
    return dict(schema='inkling2-causal-mc-v1', source_repo=recipe.BASE_REPO,
        source_revision=recipe.BASE_REVISION, source_specs=digest(source.specs),
        corpus_manifest=digest(json.loads((corpus/'manifest.json').read_text())), fork=pin,
        numerical_files={p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in NUMERICAL_FILES},
        numerical=dict(expert_bits=4, core_bits=[8, 6], upstream_core_bits=8, seed=787,
            sigma=.025, covariance=numerical.ACCUMULATION_CONTRACT,
            mtp_anchor='last-complete-block-per-untruncated-prompt', drafter_pilot=False))


class Progress:
    def __init__(self, root):
        self.root = root
        self.started = time.time()
        self.count = 0

    def emit(self, event, **values):
        if event == 'projection_committed':
            self.count += 1
        row = dict(time=datetime.now(timezone.utc).isoformat(), event=event,
            committed_encodes_this_attempt=self.count, **values)
        with (self.root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        atomic_json(self.root/'status.json', dict(status='running', phase=event, **row))
        print(json.dumps(row), flush=True)


class CachedReader:
    """Cache only native non-expert weights for the currently consuming phase."""
    def __init__(self, reader):
        self.reader, self.cache = reader, {}
        self.config = reader.config

    def tensor(self, name, **kwargs):
        if kwargs.get('expert') is not None or kwargs.get('rows') is not None:
            return self.reader.tensor(name, **kwargs)
        key = (name, str(kwargs.get('device', 'cpu')))
        if key not in self.cache:
            self.cache[key] = self.reader.tensor(name, **kwargs)
        return self.cache[key]

    def expert(self, *args, **kwargs):
        return self.reader.expert(*args, **kwargs)

    def embedding(self, *args, **kwargs):
        return self.reader.embedding(*args, **kwargs)

    def clear(self):
        self.cache.clear()


def raw(acc):
    if acc.count <= 0:
        raise ValueError('empty calibration covariance')
    return dict(H=acc.snapshot(), count=acc.count, finalized=False)


def save_stage(store, name, tensors, metadata=None):
    prior = store.load(name)
    if prior is not None:
        return prior
    directory = store.directory(name)
    directory.mkdir(parents=True, exist_ok=True)
    # A phase has no numerical standing before its complete manifest. Rewrite
    # only these task-owned batch leaves, never a committed successor.
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or not (path.name.endswith('.safetensors')
                or path.name.endswith('.safetensors.partial')) or not path.name[:5].isdigit():
            raise ValueError('unexpected uncommitted phase leaf: ' + path.name)
        path.unlink()
    entries = []
    for index, values in tensors:
        if any(not bool(torch.isfinite(v).all()) for v in values.values()):
            raise ValueError('nonfinite successor activation; do not reuse partial phase')
        filename = f'{index:05d}.safetensors'
        receipt = commit_safetensors(directory/filename, values)
        entries.append(dict(filename=filename, **receipt))
    return store.commit(name, entries, metadata)


def load_entry(store, name, index, device):
    return {k: v.to(device) for k, v in load_file(str(store.directory(name)/f'{index:05d}.safetensors')).items()}


def state_tensors(state):
    return dict(partial=state.partial, **{f'block_{i}': value for i, value in enumerate(state.blocks)})


def state_from(values, native):
    return native.AttnResState(values['partial'], [values[f'block_{i}']
        for i in range(len(values)-1)])


def source_entries(header, names):
    return {name: (header.snapshot/header.locations[name]['shard'],
                   header.locations[name]['source_name']) for name in names}


def packed_entries(store, names):
    return {name+'.'+suffix: (store.paths(name)[0], name+'.'+suffix)
            for name in names for suffix in recipe.SUFFIXES}


class Runner:
    def __init__(self, args):
        from .storage import RouteSpool
        self.RouteSpool, self.args = RouteSpool, args
        self.root, self.device = args.root.resolve(), args.device
        self.root.mkdir(parents=True, exist_ok=True)
        self.header = HeaderSource(args.source)
        self.identity = plan(self.header, args.corpus)
        identity_path = self.root/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != self.identity:
            raise ValueError('frozen numerical identity differs; diagnose before any reuse')
        atomic_json(identity_path, self.identity)
        self.progress = Progress(self.root)
        self.native = numerical.fork_module('utils.inkling2_core')
        self.reader = CachedReader(self.native.SourceReader(args.source))
        self.ref = self.native.Inkling2Reference(self.reader, device=self.device)
        self.records = list(load_token_batches(args.corpus))
        self.tokens = sum(len(self.ids(row)) for row in self.records)
        self.bounds = np.cumsum([0] + [len(self.ids(row)) for row in self.records])
        self.stages = StageStore(self.root/'activations', self.identity)
        self.taps = StageStore(self.root/'target-taps', self.identity)
        self.writer = DirectMCWriter(args.output, args.source, recipe.BASE_REPO,
            recipe.BASE_REVISION, self.identity, self.header.as_inventory())
        self.core = {bits: recipe.reviewed_core_modules(core_bits=bits) for bits in (8, 6)}
        self.experts = recipe.reviewed_expert_modules()

    @staticmethod
    def ids(row):
        return row['input_ids'].reshape(-1)

    def rows(self, stage):
        if self.stages.load(stage) is None:
            raise ValueError('missing committed activation frontier: ' + stage)
        for index, row in enumerate(self.records):
            yield index, row, load_entry(self.stages, stage, index, self.device)

    def projection_stores(self, label, selected):
        return {bits: ProjectionStore(self.root/'packed'/label/f'k{bits}',
            self.root/'errors'/f'k{bits}', self.identity,
            {name: self.core[bits][name] for name in selected}, self.progress,
            numerical, self.device) for bits in (8, 6)}

    def encode(self, stores, name, acc):
        covariance = raw(acc)
        weight = self.reader.tensor(name+'.weight', device=self.device)
        for bits in (8, 6):
            stores[bits].ensure(name, weight, covariance,
                extra_evidence=dict(upstream_core_bits=8, accumulation=numerical.ACCUMULATION_CONTRACT))
        return stores[8].dense(name)

    def hook(self, overrides):
        def apply(name, x, weight):
            replacement = overrides.get(name.removesuffix('.weight'))
            return None if replacement is None else F.linear(x.to(replacement.dtype), replacement).to(x.dtype)
        return apply

    def initialize(self):
        if (self.root/'frontier.json').exists():
            return
        if self.stages.load('boundary-000') is not None:
            atomic_json(self.root/'frontier.json', dict(next_layer=0, stage='boundary-000'))
            return
        def batches():
            for index, row in enumerate(self.records):
                yield index, state_tensors(self.ref.start(self.ids(row).to(self.device)))
        save_stage(self.stages, 'boundary-000', batches())
        atomic_json(self.root/'frontier.json', dict(next_layer=0, stage='boundary-000'))
        self.reader.clear()

    def capture_tap(self, layer, stage):
        if layer not in (5, 11, 17, 23, 29, 35, 41, 48):
            return
        name = f'tap-{layer:03d}'
        if self.taps.load(name) is None:
            def batches():
                for index, _, values in self.rows(stage):
                    state = state_from(values, self.native)
                    value = self.ref.finish(state) if layer == 48 else state.partial
                    yield index, dict(features=value)
            save_stage(self.taps, name, batches())

    def router(self, layer, x):
        ids, gamma, _ = self.ref.route(layer, x)
        prefix = f'model.llm.layers.{layer}.mlp.gate.'
        logits = self.ref.project(prefix+'weight', x, output_dtype=torch.float32)
        ranking = torch.argsort(logits.sigmoid()+self.ref.weight(prefix+'bias').float(),
                               descending=True, stable=True, dim=-1)
        near = ranking[:, 8:16]
        # Recovery substitutes each candidate for the last native route. Its
        # weight is native logsigmoid renormalization for that hypothetical set.
        chosen = F.logsigmoid(logits.gather(-1, ids[:, :7]))
        candidate = F.logsigmoid(logits.gather(-1, near))
        hypothetical = torch.cat((chosen[:, None, :].expand(-1, 8, -1), candidate[:, :, None]), -1)
        maximum = hypothetical.max(-1, keepdim=True).values
        shifted = (hypothetical-maximum).exp()
        total = ((shifted[..., 0]+shifted[..., 2])+(shifted[..., 1]+shifted[..., 3])) + ((shifted[..., 4]+shifted[..., 6])+(shifted[..., 5]+shifted[..., 7]))
        candidate_gamma = (candidate-(maximum.squeeze(-1)+total.log())).exp() * self.ref.config.get('route_scale', 8.)
        return ids, gamma, near, candidate_gamma

    def layer(self, layer, frontier):
        label, stage = f'layer-{layer:03d}', frontier['stage']
        started = time.time()
        self.capture_tap(layer, stage)
        selected = [n for n in self.core[8] if n.startswith(f'model.llm.layers.{layer}.')]
        stores = self.projection_stores(label, selected)
        overrides = {}
        entry = [n for n in selected if not n.endswith('.wo_ud')]
        missing = [n for n in entry if any(s.load(n) is None for s in stores.values())]
        if missing:
            accumulators = {n: numerical.covariance_accumulator(self.core[8][n]['in_features'], device=self.device) for n in missing}
            for index, _, values in self.rows(stage):
                inputs = self.ref.attention_inputs(state_from(values, self.native), layer)
                for name, acc in accumulators.items():
                    acc.add(inputs[name+'.weight'])
                if index % 64 == 0:
                    self.progress.emit('attention_input_capture', layer=layer, prompts=index+1)
            for name, acc in accumulators.items():
                overrides[name] = self.encode(stores, name, acc)
            del accumulators
        for name in entry:
            if name not in overrides:
                overrides[name] = stores[8].dense(name)
        output_name = next(n for n in selected if n.endswith('.wo_ud'))
        if any(s.load(output_name) is None for s in stores.values()):
            acc = numerical.covariance_accumulator(6144, device=self.device)
            for _, _, values in self.rows(stage):
                _, o_input = self.ref.attention(state_from(values, self.native), layer, self.hook(overrides))
                acc.add(o_input)
            overrides[output_name] = self.encode(stores, output_name, acc)
            del acc
        else:
            overrides[output_name] = stores[8].dense(output_name)
        attention_stage = f'attention-{layer:03d}'
        if self.stages.load(attention_stage) is None:
            def batches():
                for index, _, values in self.rows(stage):
                    state = state_from(values, self.native)
                    attention, _ = self.ref.attention(state, layer, self.hook(overrides))
                    pending, x = self.ref.attention_finish(state, attention, layer)
                    yield index, dict(attention=pending.attention, mlp_input=x)
            save_stage(self.stages, attention_stage, batches())
        del overrides
        spool = self.RouteSpool.create(self.root/'routes'/label, self.identity, self.tokens,
            self.rows(attention_stage), lambda x: self.router(layer, x), self.progress, layer,
            width=4608, experts=384, minimum=1024)
        self.reader.clear()
        expert_store = ProjectionStore(self.root/'packed'/label/'experts', self.root/'errors'/'experts',
            self.identity, self.experts, self.progress, numerical, self.device)
        outputs_path = self.root/'routes'/label/'outputs.bf16'
        output_bytes = self.tokens * 8 * 4608 * 2
        scatter_binding_path = outputs_path.with_suffix('.identity.json')
        scatter_receipts = list(outputs_path.parent.glob('expert-*.json'))
        if outputs_path.exists():
            if outputs_path.stat().st_size != output_bytes or not scatter_binding_path.exists():
                raise ValueError('scatter output extent/identity missing; diagnose before reuse')
            binding = json.loads(scatter_binding_path.read_text())
            actual = file_identity(outputs_path)
            if (binding['identity'] != self.identity or binding['route_manifest'] != digest(spool.manifest)
                    or actual['inode'] != binding['file_identity']['inode']):
                raise ValueError('scatter output or route binding differs')
        elif scatter_receipts or scatter_binding_path.exists():
            raise ValueError('scatter output missing while committed expert receipts exist')
        outputs = np.memmap(outputs_path, dtype=np.uint16, mode='r+' if outputs_path.exists() else 'w+',
                            shape=(self.tokens, 8, 4608))
        if not scatter_binding_path.exists():
            atomic_json(scatter_binding_path, dict(identity=self.identity,
                route_manifest=digest(spool.manifest), file_identity=file_identity(outputs_path)))
        try:
            for expert in range(384):
                prefix = f'model.llm.layers.{layer}.mlp.experts.{expert}.'
                gate_name, up_name, down_name = (prefix+role+'_proj' for role in ('gate', 'up', 'down'))
                receipt_path = self.root/'routes'/label/f'expert-{expert:03d}.json'
                if receipt_path.exists():
                    saved = json.loads(receipt_path.read_text())
                    if (saved.get('identity') != self.identity or saved.get('expert') != expert
                            or saved.get('scatter_inode') != outputs_path.stat().st_ino
                            or saved.get('route_manifest') != digest(spool.manifest)):
                        raise ValueError('expert scatter identity differs')
                    continue
                input_raw = None
                if any(expert_store.load(n) is None for n in (gate_name, up_name)):
                    input_raw, input_evidence = spool.hessian(expert, numerical=numerical, device=self.device)
                physical = f'model.llm.layers.{layer}.mlp.experts.'
                for name, role in ((gate_name, 'gate'), (up_name, 'up')):
                    if expert_store.load(name) is None:
                        weight = self.reader.expert(physical+'w13_weight', expert, projection=role, device=self.device)
                        expert_store.ensure(name, weight, input_raw, extra_evidence=input_evidence)
                gate, up = expert_store.dense(gate_name), expert_store.dense(up_name)
                def activated(x, gamma):
                    g, u = F.linear(x, gate), F.linear(x, up)
                    fused = torch.stack((g, u), -1).flatten(-2)
                    return numerical.mxfp8_qdq(self.native.capped_swiglu(fused, gamma,
                        self.ref.config.get('mlp_gate_softcap', 4.), self.ref.config.get('mlp_up_softcap', 25.)))
                if expert_store.load(down_name) is None:
                    down_raw, down_evidence = spool.hessian(expert, down=activated, down_width=2304,
                        numerical=numerical, device=self.device)
                    weight = self.reader.expert(physical+'w2_weight', expert, projection='down', device=self.device)
                    expert_store.ensure(down_name, weight, down_raw, extra_evidence=down_evidence)
                    del down_raw, weight
                down = expert_store.dense(down_name)
                natural, _ = spool.rows(expert)
                for start in range(0, len(natural), 2048):
                    rows = natural[start:start+2048]
                    x = numerical.mxfp8_qdq(spool.gather(rows, self.device))
                    gamma = spool.gamma(expert, rows, device=self.device)
                    result = F.linear(activated(x, gamma), down)
                    routes = np.asarray(spool.routes[rows])
                    rank = np.argmax(routes == expert, axis=1)
                    outputs[rows, rank] = result.cpu().contiguous().view(torch.uint16).numpy()
                outputs.flush()
                with outputs_path.open('rb') as stream:
                    os.fsync(stream.fileno())
                atomic_json(receipt_path, dict(identity=self.identity, expert=expert, natural_rows=len(natural),
                    scatter_inode=outputs_path.stat().st_ino, route_manifest=digest(spool.manifest)))
                del gate, up, down, input_raw
        finally:
            outputs.flush()
            outputs._mmap.close()
        next_stage = f'boundary-{layer+1:03d}'
        if self.stages.load(next_stage) is None:
            mmap = np.memmap(outputs_path, dtype=np.uint16, mode='r', shape=(self.tokens, 8, 4608))
            try:
                def batches():
                    for index, _, values in self.rows(stage):
                        attention = load_entry(self.stages, attention_stage, index, self.device)['attention']
                        parts = torch.from_numpy(np.array(mmap[self.bounds[index]:self.bounds[index+1]], copy=True)).view(torch.bfloat16).to(self.device)
                        result = torch.zeros((len(parts), 4608), dtype=torch.float32, device=self.device)
                        for rank in range(8):
                            result = result + parts[:, rank].float()
                        pending = self.native.PendingLayer(state_from(values, self.native), attention, layer)
                        yield index, state_tensors(self.ref.mlp_finish(pending, result.to(torch.bfloat16)))
                save_stage(self.stages, next_stage, batches())
            finally:
                mmap._mmap.close()
        original_names = {n+'.weight' for n in selected}
        native_names = {n for n in self.header.specs if n.startswith(f'model.llm.layers.{layer}.')
            and n not in original_names and '.mlp.experts.' not in n}
        self.writer.write_group(label, shared=source_entries(self.header, native_names) |
            packed_entries(expert_store, [n for n in self.experts if n.startswith(f'model.llm.layers.{layer}.')]),
            original=source_entries(self.header, original_names),
            k8=packed_entries(stores[8], selected), k6=packed_entries(stores[6], selected))
        atomic_json(self.root/'frontier.json', dict(next_layer=layer+1, stage=next_stage))
        atomic_json(self.root/'layers'/label/'complete.json', dict(identity=self.identity,
            layer=layer, seconds=time.time()-started, encode_operations=1164))
        self.stages.retire(stage, attention_stage)
        for store in (*stores.values(), expert_store):
            store.retire()
        spool.close()
        shutil.rmtree(self.root/'routes'/label)
        self.reader.clear()
        self.progress.emit('layer_complete', layer=layer, seconds=time.time()-started)

    def target_finish(self):
        frontier = json.loads((self.root/'frontier.json').read_text())
        self.capture_tap(48, frontier['stage'])
        name = 'model.llm.unembed'
        stores = self.projection_stores('head', [name])
        if any(s.load(name) is None for s in stores.values()):
            acc = numerical.covariance_accumulator(4608, device=self.device)
            for _, _, values in self.rows(frontier['stage']):
                acc.add(self.ref.finish(state_from(values, self.native)))
            self.encode(stores, name, acc)
        self.writer.write_group('head', original=source_entries(self.header, [name+'.weight']),
            k8=packed_entries(stores[8], [name]), k6=packed_entries(stores[6], [name]))
        for store in stores.values():
            store.retire()
        self.reader.clear()

    def features(self, index, anchor):
        return torch.stack([load_entry(self.taps, f'tap-{n:03d}', index, self.device)['features'][:anchor]
                            for n in (5, 11, 17, 23, 29, 35, 41, 48)], dim=1)

    def drafter(self):
        for tap in (5, 11, 17, 23, 29, 35, 41, 48):
            if self.taps.load(f'tap-{tap:03d}') is None:
                raise ValueError('missing committed K8 target tap')
        stage = 'mtp-boundary-000'
        if self.stages.load(stage) is None:
            def batches():
                for index, row in enumerate(self.records):
                    ids = self.ids(row).to(self.device)
                    anchor = len(ids)-8
                    yield index, dict(block=self.ref.mtp_embed(ids[anchor:anchor+1])[0])
            save_stage(self.stages, stage, batches())
        for layer in range(5):
            label = f'mtp-{layer:03d}'
            selected = [n for n in self.core[8] if n.startswith(f'mtp.layers.{layer}.')]
            next_stage = f'mtp-boundary-{layer+1:03d}'
            if self.writer.group_complete(label) and self.stages.load(next_stage) is not None:
                stage = next_stage
                continue
            stores, overrides = self.projection_stores(label, selected), {}
            groups = ([n for n in selected if n.endswith(('.q_proj', '.k_proj', '.v_proj', '.k_proj_ctx', '.v_proj_ctx'))],
                      [n for n in selected if n.endswith('.o_proj')],
                      [n for n in selected if n.endswith(('.gate_proj', '.up_proj'))],
                      [n for n in selected if n.endswith('.down_proj')])
            for group in groups:
                missing = [n for n in group if any(s.load(n) is None for s in stores.values())]
                if missing:
                    accs = {n: numerical.covariance_accumulator(self.core[8][n]['in_features'], device=self.device) for n in missing}
                    def capture(name, x, weight):
                        canonical_name = name.removesuffix('.weight')
                        if canonical_name in accs:
                            accs[canonical_name].add(x)
                        return self.hook(overrides)(name, x, weight)
                    for index, row in enumerate(self.records):
                        block = load_entry(self.stages, stage, index, self.device)['block']
                        anchor = len(self.ids(row))-8
                        self.ref.mtp_layer(block, self.features(index, anchor), layer, anchor, hook=capture)
                    for name, acc in accs.items():
                        overrides[name] = self.encode(stores, name, acc)
                    del accs
                for name in group:
                    if name not in overrides:
                        overrides[name] = stores[8].dense(name)
            if self.stages.load(next_stage) is None:
                def batches():
                    for index, row in enumerate(self.records):
                        block = load_entry(self.stages, stage, index, self.device)['block']
                        anchor = len(self.ids(row))-8
                        yield index, dict(block=self.ref.mtp_layer(block, self.features(index, anchor), layer, anchor, hook=self.hook(overrides)))
                save_stage(self.stages, next_stage, batches())
            self.writer.write_group(label, original=source_entries(self.header, [n+'.weight' for n in selected]),
                k8=packed_entries(stores[8], selected), k6=packed_entries(stores[6], selected))
            for store in stores.values():
                store.retire()
            del overrides
            self.reader.clear()
            stage = next_stage
            self.progress.emit('drafter_layer_complete', layer=layer)

    def seal(self):
        selected = self.core[8]
        exported = {n for group in self.writer.state['groups'].values()
                    for kind in ('shared', 'original') for n in group['entries'][kind]}
        remaining = {n for n in self.header.specs if '.mlp.experts.' not in n}-exported
        if any('.mlp.experts.' in n or n.removesuffix('.weight') in selected for n in remaining):
            raise ValueError('missing encoded source inventory before package seal')
        if not self.writer.group_complete('native'):
            self.writer.write_group('native', shared=source_entries(self.header, remaining))
        configs, quantizes = {}, {}
        dtype_names = {'I16':'int16', 'F16':'float16', 'I32':'int32'}
        native = {n: dict(dtype=dtype, shape=shape) for n, (dtype, shape) in self.header.specs.items()
            if '.mlp.experts.' not in n and n.removesuffix('.weight') not in selected}
        for view, bits in (('original', None), ('k8-compact', 8), ('k6-compact', 6)):
            config = deepcopy(self.header.document)
            quantize_file = {'original':'quantize_config.json', 'k8-compact':'quantize_config.k8.json',
                             'k6-compact':'quantize_config.k6.json'}[view]
            metadata_pointer = quantize_file+':tensor_storage'
            config['quantization_config'] = dict(quant_method='exl3', format='mcg',
                per_module_metadata=metadata_pointer, activation_quantization=recipe.ACTIVATION_QUANTIZATION)
            modules = self.experts | ({} if bits is None else self.core[bits])
            storage = {name: dict(bits_per_weight=g['bits'], stored_tensors={n: dict(shape=shape,
                torch_dtype=dtype_names[dtype]) for n, (dtype, shape) in recipe.packed_specs(name, g).items()})
                for name, g in modules.items()}
            configs[view] = config
            quantizes[view] = dict(quant_method='exl3', tensor_storage=storage,
                per_module_metadata=metadata_pointer,
                retained_native_tensors=native | ({} if bits else {
                    n+'.weight': dict(dtype='BF16', shape=[g['out_features'], g['in_features']])
                    for n,g in selected.items()}), provenance=self.identity,
                core_view=view, upstream_activations='k8-compact')
        assets = {p.name:p for p in self.header.snapshot.iterdir() if p.is_file()
            and p.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                'generation_config.json', 'special_tokens_map.json', 'preprocessor_config.json',
                'processor_config.json')}
        receipt = self.writer.finalize(core_selection=selected, configs=configs,
            quantizes=quantizes, assets=assets, expected_payload_bytes=recipe.TARGET_PAYLOAD_BYTES)
        atomic_json(self.root/'quantization-complete.json', dict(status='passed', identity=self.identity,
            package_validation=receipt, output=str(self.args.output)))
        self.progress.emit('quantization_complete', layers=48, encode_operations=recipe.ENCODE_COUNT)

    def run(self):
        if (self.root/'quantization-complete.json').exists():
            receipt = json.loads((self.root/'quantization-complete.json').read_text())
            if receipt.get('status') != 'passed' or receipt.get('identity') != self.identity:
                raise ValueError('unrecognized completed-job identity; diagnose without restarting')
            return
        self.initialize()
        while True:
            frontier = json.loads((self.root/'frontier.json').read_text())
            if frontier['next_layer'] == 48:
                break
            self.layer(frontier['next_layer'], frontier)
        if not self.writer.group_complete('head'):
            self.target_finish()
        self.drafter()
        self.seal()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.90, torch.device(args.device))
    Runner(args).run()


if __name__ == '__main__':
    main()
